FROM mcr.microsoft.com/devcontainers/universal:2

# Remove stale Yarn apt source entries that can break apt-get update in feature installs.
RUN set -eux; \
		rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn*.list; \
		if [ -d /etc/apt/sources.list.d ]; then \
			find /etc/apt/sources.list.d -type f -name '*.list' -exec sed -i '/dl\.yarnpkg\.com/d' {} +; \
		fi

WORKDIR /workspaces/hycu-project
