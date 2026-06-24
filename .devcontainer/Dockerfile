# FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel
FROM python:3.11-slim

ARG TZ
ARG CLAUDE_CODE_VERSION=latest
ARG OPENCODE_VERSION=latest
ARG GEMINI_CLI_VERSION=latest
ARG GIT_DELTA_VERSION=0.18.2
ARG ZSH_IN_DOCKER_VERSION=1.2.1

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano
ENV PIP_NO_CACHE_DIR=1

# Install system packages and Node.js (single RUN to reduce layers)
# Install basic development tools, Python 3.11, and dependencies
RUN rm -rf /var/lib/apt/lists/* && apt-get update --fix-missing \
  && apt-get install -y --no-install-recommends \
  curl \
  wget \
  gnupg2 \
  && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
  && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
  less \
  git \
  procps \
  sudo \
  fzf \
  zsh \
  man-db \
  unzip \
  gh \
  iptables \
  ipset \
  iproute2 \
  dnsutils \
  aggregate \
  jq \
  nano \
  vim \
  tmux \
  libgl1 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libsdl2-dev \
  libsdl2-mixer-dev \
  libsdl2-image-dev \
  libsdl2-ttf-dev \
  libfreetype6-dev \
  libpng-dev \
  libjpeg-dev \
  libtiff-dev \
  graphviz \
  pkg-config \
  libffi-dev \
  libssl-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install audio packages
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  libportmidi-dev \
  fluidsynth \
  timidity \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js 20
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
apt-get install -y nodejs

# Create non-root user with npm global setup
RUN groupadd --gid 1000 node \
  && useradd --uid 1000 --gid node --shell /bin/bash --create-home node \
  && mkdir -p /usr/local/share/npm-global /workspace \
  && chown -R node:node /workspace /usr/local/share/npm-global

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin

# Persist bash history.
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R node /commandhistory

RUN su node -c "sh -c \"\$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v\${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)\" -- \
  -p git \
  -p fzf \
  -a \"source /usr/share/doc/fzf/examples/key-bindings.zsh\" \
  -a \"source /usr/share/doc/fzf/examples/completion.zsh\" \
  -a \"export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history\" \
  -x"

USER root
# Create Claude config directory and set ownership
RUN mkdir -p /home/node/workspace/.claude && chown -R node:node /home/node/workspace/.claude
ENV CLAUDE_CONFIG_DIR="/home/node/workspace/.claude"

# Install Claude CLI globally as node user
USER node
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} \
    && npm install -g opencode-ai@${OPENCODE_VERSION} \
    && npm install -g @google/gemini-cli@${GEMINI_CLI_VERSION}
RUN git config --global --add safe.directory /workspace
ENV NODE_OPTIONS="--max-old-space-size=4096" CLAUDE_CONFIG_DIR="/home/node/workspace/.claude" POWERLEVEL9K_DISABLE_GITSTATUS="true"
ENV PATH="$PATH:/home/node/.local/bin"

WORKDIR /workspace
    
# Install Python packages from pyproject.toml
COPY pyproject.toml README.md ./
COPY src ./src

USER node
RUN pip install --user -U pip \
    && pip install --user --no-cache-dir -e ".[dev]" \
    && pip install --user --no-cache-dir jupyter jupyter-core typing_extensions

USER root
# To run jupyter in remote development scenariowith VSCode
# from https://stackoverflow.com/questions/63998873/vscode-how-to-run-a-jupyter-notebook-in-a-docker-container-over-a-remote-serve
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION=v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini


# Configure Jupyter
RUN mkdir -p /home/node/.jupyter \
  && echo "c.ServerApp.ip = '0.0.0.0'" > /home/node/.jupyter/jupyter_lab_config.py \
  && echo "c.ServerApp.allow_root = True" >> /home/node/.jupyter/jupyter_lab_config.py \
  && echo "c.ServerApp.open_browser = False" >> /home/node/.jupyter/jupyter_lab_config.py \
  && echo "c.IdentityProvider.token = ''" >> /home/node/.jupyter/jupyter_lab_config.py \
  && chown -R node:node /home/node/.jupyter

# Expose ports for Dash app and Jupyter
EXPOSE  8888

# Default command (overridden by docker-compose with sleep infinity)
# Jupyter is started via postStartCommand in devcontainer.json
CMD ["tini", "--", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]