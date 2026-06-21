FROM mcr.microsoft.com/devcontainers/universal:2

# Remove stale Yarn apt source entries that can break apt-get update in feature installs.
RUN set -eux; \
		rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn*.list; \
		if [ -d /etc/apt/sources.list.d ]; then \
			find /etc/apt/sources.list.d -type f -name '*.list' -exec sed -i '/dl\.yarnpkg\.com/d' {} +; \
		fi

# Markdown publishing toolchain used by scripts/render_documents.py.
# Mermaid CLI downloads its headless browser into a shared, readable cache so
# the non-root devcontainer user can execute mmdc after the image is built.
ENV PUPPETEER_CACHE_DIR=/opt/puppeteer
RUN set -eux; \
		apt-get update; \
		DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
			default-jre-headless \
			fonts-dejavu-core \
			fonts-liberation \
			pandoc \
			plantuml; \
		rm -rf /var/lib/apt/lists/*; \
		mkdir -p "$PUPPETEER_CACHE_DIR"; \
		npm install --global @mermaid-js/mermaid-cli; \
		chmod -R a+rX "$PUPPETEER_CACHE_DIR"

WORKDIR /workspaces/hycu-project
