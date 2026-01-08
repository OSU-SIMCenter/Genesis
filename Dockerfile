# syntax=docker/dockerfile:1

# --- STAGE 1: BUILDER ---
FROM ghcr.io/prefix-dev/pixi:jammy AS builder

WORKDIR /app

# 1. Install dependencies (leveraging cache)
COPY pixi.toml pixi.lock ./
RUN pixi install --locked

# 2. Copy source and build
COPY . .
# Using build.py for consistency. 
# --target=local inside the container means we build FOR the container's OS (Linux).
RUN pixi run python build.py --target=local --tool=nuitka

# --- STAGE 2: RUNTIME ---
FROM gcr.io/distroless/cc-debian12

WORKDIR /app

# Copy artifact from builder
COPY --from=builder /app/dist/nuitka/linux/teleop_socket.dist /app

# Set library path for contained libs
ENV LD_LIBRARY_PATH=/app

EXPOSE 8765

ENTRYPOINT ["/app/teleop_socket"]
