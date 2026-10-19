//! Responses の状態保存 (previous_response_id、GET / DELETE / input_items)。
//! sqlite (WAL、本文は zstd) とプロセス内の LRU (r7_proxy_spec.md §6.6)。

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};

use crate::config::State as Cfg;
use crate::util::now_unix;

/// 1 応答分の保存内容。
#[derive(Debug, Clone)]
pub struct Stored {
    pub id: String,
    pub created: u64,
    pub previous_response_id: Option<String>,
    /// 応答オブジェクト全体。
    pub response: Value,
    /// この要求の input を展開した項目 (item_reference 解決済み、id 付き)。
    pub input_items: Vec<Value>,
}

impl Stored {
    fn to_raw(&self) -> Vec<u8> {
        let v = json!({"id": self.id, "created": self.created, "prev": self.previous_response_id,
            "response": self.response, "input": self.input_items});
        serde_json::to_vec(&v).unwrap_or_default()
    }
    fn compress(raw: Vec<u8>) -> Vec<u8> {
        zstd::bulk::compress(&raw, 3).unwrap_or(raw)
    }
    /// (中身, 展開後の JSON のバイト数)。
    fn from_blob(b: &[u8]) -> Option<(Self, usize)> {
        let raw = zstd::stream::decode_all(b).unwrap_or_else(|_| b.to_vec());
        let size = raw.len();
        let v: Value = serde_json::from_slice(&raw).ok()?;
        Some((Self {
            id: v.get("id")?.as_str()?.to_string(),
            created: v.get("created").and_then(Value::as_u64).unwrap_or(0),
            previous_response_id: v.get("prev").and_then(Value::as_str).map(str::to_string),
            response: v.get("response").cloned().unwrap_or(Value::Null),
            input_items: v.get("input").and_then(Value::as_array).cloned().unwrap_or_default(),
        }, size))
    }
    /// 出力の項目 (次のターンの入力に使う)。
    pub fn output_items(&self) -> Vec<Value> {
        self.response.get("output").and_then(Value::as_array).cloned().unwrap_or_default()
    }
}

/// プロセス内 LRU: (id → (中身, バイト数), 古い順の id, 合計バイト数)。
type Lru = (HashMap<String, (Arc<Stored>, usize)>, VecDeque<String>, usize);

pub struct Store {
    db: Option<Arc<Mutex<Connection>>>,
    lru: Mutex<Lru>,
    memory_items: usize,
    memory_bytes: usize,
    ttl_s: u64,
    max_bytes: u64,
}

impl Store {
    pub fn open(cfg: &Cfg) -> Self {
        let ttl_s = (cfg.ttl_days * 86_400.0) as u64;
        let max_bytes = (cfg.max_gb * (1u64 << 30) as f64) as u64;
        let mut db = None;
        if cfg.backend == "sqlite" {
            match open_db(&cfg.path) {
                Ok(c) => db = Some(Arc::new(Mutex::new(c))),
                Err(e) => eprintln!("ds41gw: sqlite {} を開けない ({e})。メモリだけで保存する", cfg.path.display()),
            }
        }
        Self {
            db,
            lru: Mutex::new((HashMap::new(), VecDeque::new(), 0)),
            memory_items: cfg.memory_items.max(1),
            memory_bytes: cfg.memory_mb.max(1).saturating_mul(1 << 20),
            ttl_s,
            max_bytes,
        }
    }

    pub fn memory(cfg: &Cfg) -> Self {
        let mut c = cfg.clone();
        c.backend = "memory".into();
        Self::open(&c)
    }

    fn lru_put(&self, s: Arc<Stored>, size: usize) {
        let mut g = self.lru.lock().unwrap();
        match g.0.insert(s.id.clone(), (s.clone(), size)) {
            Some((_, old)) => g.2 = g.2.saturating_sub(old),
            None => g.1.push_back(s.id.clone()),
        }
        g.2 += size;
        // 件数とバイト数の両方で抑える (いま入れたものは残す)
        while g.1.len() > self.memory_items || (g.2 > self.memory_bytes && g.1.len() > 1) {
            let Some(k) = g.1.pop_front() else { break };
            if let Some((_, sz)) = g.0.remove(&k) {
                g.2 = g.2.saturating_sub(sz);
            }
        }
        // 1 件で上限を超えるものはメモリに置かない (sqlite にはある)
        if g.2 > self.memory_bytes && self.db.is_some() {
            if let Some(k) = g.1.pop_front() {
                if let Some((_, sz)) = g.0.remove(&k) {
                    g.2 = g.2.saturating_sub(sz);
                }
            }
        }
    }

    pub fn memory_usage(&self) -> (usize, usize) {
        let g = self.lru.lock().unwrap();
        (g.0.len(), g.2)
    }

    pub async fn put(&self, s: Stored) {
        let s = Arc::new(s);
        let raw = s.to_raw();
        self.lru_put(s.clone(), raw.len());
        if let Some(db) = &self.db {
            let db = db.clone();
            let (id, created, prev) = (s.id.clone(), s.created as i64, s.previous_response_id.clone());
            let _ = tokio::task::spawn_blocking(move || {
                let blob = Stored::compress(raw);
                let c = db.lock().unwrap();
                c.execute(
                    "INSERT OR REPLACE INTO responses (id, created, prev, size, data) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![id, created, prev, blob.len() as i64, blob],
                )
            })
            .await;
        }
    }

    pub async fn get(&self, id: &str) -> Option<Arc<Stored>> {
        let hit = self.lru.lock().unwrap().0.get(id).map(|(s, _)| s.clone());
        let s = match hit {
            Some(s) => s,
            None => {
                let db = self.db.clone()?;
                let id2 = id.to_string();
                let blob: Option<Vec<u8>> = tokio::task::spawn_blocking(move || {
                    let c = db.lock().unwrap();
                    c.query_row("SELECT data FROM responses WHERE id = ?1", params![id2], |r| r.get(0)).optional().ok().flatten()
                })
                .await
                .ok()
                .flatten();
                let (s, size) = Stored::from_blob(&blob?)?;
                let s = Arc::new(s);
                self.lru_put(s.clone(), size);
                s
            }
        };
        if self.ttl_s > 0 && now_unix().saturating_sub(s.created) > self.ttl_s {
            return None;
        }
        Some(s)
    }

    pub async fn delete(&self, id: &str) -> bool {
        let mut found = {
            let mut g = self.lru.lock().unwrap();
            let had = match g.0.remove(id) {
                Some((_, sz)) => {
                    g.2 = g.2.saturating_sub(sz);
                    true
                }
                None => false,
            };
            g.1.retain(|k| k != id);
            had
        };
        if let Some(db) = &self.db {
            let db = db.clone();
            let id2 = id.to_string();
            let n = tokio::task::spawn_blocking(move || {
                let c = db.lock().unwrap();
                c.execute("DELETE FROM responses WHERE id = ?1", params![id2]).unwrap_or(0)
            })
            .await
            .unwrap_or(0);
            found |= n > 0;
        }
        found
    }

    /// 連鎖を親までたどり、[親の input, 親の output, …] を返す (§6.6)。
    pub async fn history(&self, id: &str) -> Result<Vec<Value>, String> {
        let mut chain = Vec::new();
        let mut cur = Some(id.to_string());
        while let Some(i) = cur {
            if chain.len() >= 1000 {
                return Err("previous_response_id chain is too deep".into());
            }
            let Some(s) = self.get(&i).await else {
                return Err(format!("Previous response with id '{i}' not found."));
            };
            cur = s.previous_response_id.clone();
            chain.push(s);
        }
        let mut items = Vec::new();
        for s in chain.iter().rev() {
            items.extend(s.input_items.iter().cloned());
            items.extend(s.output_items());
        }
        Ok(items)
    }

    /// 期限切れと容量超過の掃除 (1 時間ごと)。
    pub fn sweep(&self) {
        let Some(db) = &self.db else { return };
        let c = db.lock().unwrap();
        if self.ttl_s > 0 {
            let cutoff = now_unix().saturating_sub(self.ttl_s) as i64;
            let _ = c.execute("DELETE FROM responses WHERE created < ?1", params![cutoff]);
        }
        let total: i64 = c.query_row("SELECT COALESCE(SUM(size),0) FROM responses", [], |r| r.get(0)).unwrap_or(0);
        if total as u64 > self.max_bytes {
            let mut excess = total as u64 - self.max_bytes;
            let mut stmt = match c.prepare("SELECT id, size FROM responses ORDER BY created ASC") {
                Ok(s) => s,
                Err(_) => return,
            };
            let rows: Vec<(String, i64)> = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
                .map(|it| it.flatten().collect())
                .unwrap_or_default();
            for (id, size) in rows {
                if excess == 0 {
                    break;
                }
                let _ = c.execute("DELETE FROM responses WHERE id = ?1", params![id]);
                excess = excess.saturating_sub(size as u64);
            }
        }
        let _ = c.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
    }

    pub fn count(&self) -> usize {
        if let Some(db) = &self.db {
            let c = db.lock().unwrap();
            return c.query_row("SELECT COUNT(*) FROM responses", [], |r| r.get::<_, i64>(0)).unwrap_or(0) as usize;
        }
        self.lru.lock().unwrap().0.len()
    }
}

fn open_db(path: &Path) -> rusqlite::Result<Connection> {
    if let Some(dir) = path.parent() {
        let _ = crate::reqlog::create_private_dir(dir);
    }
    let c = Connection::open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    c.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
         CREATE TABLE IF NOT EXISTS responses (
           id TEXT PRIMARY KEY, created INTEGER NOT NULL, prev TEXT, size INTEGER NOT NULL, data BLOB NOT NULL);
         CREATE INDEX IF NOT EXISTS responses_created ON responses(created);",
    )?;
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(t: &str) -> Value {
        json!({"type": "message", "role": "user", "content": t})
    }

    #[tokio::test]
    async fn lru_is_bounded_by_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Cfg { path: dir.path().join("r.sqlite"), memory_mb: 1, ..Cfg::default() };
        let st = Store::open(&cfg);
        let big = "A".repeat(300_000);
        for i in 0..10 {
            st.put(Stored {
                id: format!("resp_{i}"),
                created: now_unix(),
                previous_response_id: None,
                response: json!({"id": format!("resp_{i}"), "output": []}),
                input_items: vec![json!({"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": format!("data:image/png;base64,{big}")}]})],
            })
            .await;
        }
        let (n, bytes) = st.memory_usage();
        assert!(bytes <= 1 << 20, "lru holds {bytes} bytes in {n} items");
        assert!(n >= 1);
        // 追い出されたものも sqlite から引ける
        assert!(st.get("resp_0").await.is_some());
        assert!(st.memory_usage().1 <= 1 << 20);
    }

    #[tokio::test]
    async fn chain_delete_ttl_sqlite() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Cfg { path: dir.path().join("r.sqlite"), ..Cfg::default() };
        let st = Store::open(&cfg);
        let mk = |id: &str, prev: Option<&str>, inp: &str, out: &str, created: u64| Stored {
            id: id.into(),
            created,
            previous_response_id: prev.map(str::to_string),
            response: json!({"id": id, "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": out}]}]}),
            input_items: vec![item(inp)],
        };
        let now = now_unix();
        st.put(mk("resp_1", None, "a", "A", now)).await;
        st.put(mk("resp_2", Some("resp_1"), "b", "B", now)).await;
        st.put(mk("resp_3", Some("resp_2"), "c", "C", now)).await;
        let h = st.history("resp_3").await.unwrap();
        assert_eq!(h.len(), 6);
        assert_eq!(h[0]["content"], "a");
        assert_eq!(h[5]["content"][0]["text"], "C");
        // LRU を飛ばして sqlite から読めるか
        let st2 = Store::open(&cfg);
        assert_eq!(st2.history("resp_3").await.unwrap().len(), 6);
        assert!(st2.delete("resp_2").await);
        assert!(st2.history("resp_3").await.is_err());
        assert!(!st2.delete("resp_2").await);
        // 期限切れ
        st2.put(mk("resp_old", None, "x", "X", now - 40 * 86_400)).await;
        assert!(st2.get("resp_old").await.is_none());
        st2.sweep();
        assert_eq!(st2.count(), 2);
    }
}
