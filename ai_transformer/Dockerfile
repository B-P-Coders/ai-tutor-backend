FROM alpine:latest AS base
RUN apk add openssl
WORKDIR /app/

FROM rust:1.73-alpine AS build
RUN apk add musl-dev clang15 clang15-dev clang15-libclang openssl-dev wget git cmake
RUN rustup component add rustfmt
WORKDIR /src/
COPY . .
RUN cargo build --release

FROM base AS final
WORKDIR /app/
COPY --from=build /src/target/release/ai_transformer /app/ai_transformer
CMD [ "/app/ai_transformer" ]
