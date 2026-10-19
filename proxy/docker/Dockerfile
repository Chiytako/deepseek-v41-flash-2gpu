# ds41gw: 多段ビルド。recipe は edition 2024 と let-chains を使うので Rust 1.88 以上が要る。
# ビルド: sudo docker build -t ds41gw:0.1.0 -f docker/Dockerfile .   (文脈は proxy/)
FROM rust:1.90-bookworm AS build
WORKDIR /src
# 数値 effort 対応版の recipe (Cargo.toml の [patch.crates-io] が ../vendor を指す)
COPY vendor /vendor
# 依存だけ先にビルドして層を分ける (BuildKit が無い環境でも 2 回目以降が速い)
COPY ds41gw/Cargo.toml ds41gw/Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && touch src/lib.rs \
    && cargo build --release --locked && rm -rf src target/release/deps/ds41gw* target/release/ds41gw* target/release/.fingerprint/ds41gw*
COPY ds41gw/src ./src
RUN touch src/lib.rs src/main.rs && cargo build --release --locked && cp target/release/ds41gw /ds41gw

FROM debian:bookworm-slim
COPY --from=build /ds41gw /usr/local/bin/ds41gw
USER 1000:1000
EXPOSE 30001
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/ds41gw", "--probe", "http://127.0.0.1:30001/health"]
ENTRYPOINT ["/usr/local/bin/ds41gw"]
CMD ["--config", "/etc/ds41gw/ds41gw.toml"]
