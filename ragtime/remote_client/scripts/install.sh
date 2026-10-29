#!/bin/sh
# Ragtime Remote Client installer. Run this script with a short-lived enrollment token.
set -eu

PROGRAM=ragtime-remote
TAG_PREFIX='# ragtime-remote:'
MODE=auto
SERVER_URL=
TOKEN=
SSHD_PORT=
CA_FILE=
TUNNEL_CLIENT=
OFFLINE_PAIRING_ID=
OFFLINE_PUBLIC_KEY=
UNINSTALL=0
TUNNEL_PREREQUISITE_ERROR=

usage() {
  cat <<'EOF'
Usage: install.sh --server-url HTTPS_URL --token TOKEN [--mode auto|direct|tunnel]
                  [--port PORT] [--ca-file FILE] [--tunnel-client FILE_OR_COMMAND] [--uninstall PAIRING_ID]
       install.sh --mode offline --pairing-id ID --public-key 'ssh-ed25519 ...' [--port PORT]

The server URL must use HTTPS. --ca-file adds a private CA for the enrollment
request; TLS verification is never disabled. Offline mode prints an enrollment
blob and does not contact the server.
EOF
}

die() { printf '%s\n' "$*" >&2; exit 1; }
need_value() { [ "$#" -gt 1 ] || die "Missing value for $1"; }
json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/\\t/g' | tr '\n' ' '; }
valid_port() { case "$1" in ''|*[!0-9]*) return 1;; esac; [ "$1" -ge 1 ] 2>/dev/null && [ "$1" -le 65535 ] 2>/dev/null; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --server-url) need_value "$@"; SERVER_URL=$2; shift 2;;
    --token) need_value "$@"; TOKEN=$2; shift 2;;
    --mode) need_value "$@"; MODE=$2; shift 2;;
    --port) need_value "$@"; SSHD_PORT=$2; shift 2;;
    --ca-file) need_value "$@"; CA_FILE=$2; shift 2;;
    --tunnel-client) need_value "$@"; TUNNEL_CLIENT=$2; shift 2;;
    --pairing-id) need_value "$@"; OFFLINE_PAIRING_ID=$2; shift 2;;
    --public-key) need_value "$@"; OFFLINE_PUBLIC_KEY=$2; shift 2;;
    --uninstall) need_value "$@"; UNINSTALL=1; PAIRING_ID=$2; shift 2;;
    -h|--help) usage; exit 0;;
    *) die "Unknown option: $1";;
  esac
done

[ "$UNINSTALL" -eq 1 ] || {
  case "$MODE" in auto|direct|tunnel|offline) ;; *) die 'Mode must be auto, direct, tunnel, or offline';; esac
  if [ "$MODE" = offline ]; then
    [ -n "$OFFLINE_PAIRING_ID" ] && [ -n "$OFFLINE_PUBLIC_KEY" ] || die 'Offline mode requires --pairing-id and --public-key'
  else
    [ -n "$TOKEN" ] || die '--token is required'
    case "$SERVER_URL" in https://*) ;; *) die '--server-url must be an HTTPS URL';; esac
  fi
  [ -z "$CA_FILE" ] || [ -r "$CA_FILE" ] || die "CA file is not readable: $CA_FILE"
}

TARGET_USER=${SUDO_USER:-$(id -un)}
if [ -z "${SUDO_USER:-}" ]; then TARGET_HOME=$HOME
elif command -v getent >/dev/null 2>&1; then TARGET_HOME=$(getent passwd "$TARGET_USER" | awk -F: 'NR == 1 {print $6}')
elif command -v dscl >/dev/null 2>&1; then TARGET_HOME=$(dscl . -read "/Users/$TARGET_USER" NFSHomeDirectory 2>/dev/null | awk 'NR == 1 {print $2}')
else TARGET_HOME=; fi
[ -n "$TARGET_HOME" ] && [ -d "$TARGET_HOME" ] || die "Cannot determine home directory for $TARGET_USER"
SSH_DIR=$TARGET_HOME/.ssh
KEY_FILE=$SSH_DIR/authorized_keys
STATE_DIR=${XDG_STATE_HOME:-$TARGET_HOME/.local/state}/$PROGRAM

remove_owned() {
  [ -f "$KEY_FILE" ] || return 0
  tmp=$(mktemp "${TMPDIR:-/tmp}/ragtime-keys.XXXXXX")
  grep -Fv "${TAG_PREFIX}$PAIRING_ID" "$KEY_FILE" > "$tmp" || true
  cat "$tmp" > "$KEY_FILE"
  rm -f "$tmp"
  chmod 600 "$KEY_FILE"
  rm -rf "${STATE_DIR:?}/${PAIRING_ID:?}"
  if [ "$(id -u)" -eq 0 ] && command -v systemctl >/dev/null 2>&1; then
    systemctl disable --now "ragtime-remote-$PAIRING_ID.service" 2>/dev/null || true
    rm -f "/etc/systemd/system/ragtime-remote-$PAIRING_ID.service"
    systemctl daemon-reload 2>/dev/null || true
  fi
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user disable --now "ragtime-remote-$PAIRING_ID.service" 2>/dev/null || true
    rm -f "$TARGET_HOME/.config/systemd/user/ragtime-remote-$PAIRING_ID.service"
    systemctl --user daemon-reload 2>/dev/null || true
  fi
  if [ "$(id -u)" -eq 0 ] && command -v rc-service >/dev/null 2>&1; then
    rc-service "ragtime-remote-$PAIRING_ID" stop 2>/dev/null || true
    rc-update del "ragtime-remote-$PAIRING_ID" default 2>/dev/null || true
    rm -f "/etc/init.d/ragtime-remote-$PAIRING_ID"
  fi
  if [ "$(uname)" = Darwin ]; then
    plist="$TARGET_HOME/Library/LaunchAgents/ragtime-remote-$PAIRING_ID.plist"
    launchctl bootout "gui/$(id -u "$TARGET_USER")" "$plist" 2>/dev/null || true
    rm -f "$plist"
  fi
  printf 'Removed Ragtime key and configuration owned by pairing %s. Existing sshd was not changed.\n' "$PAIRING_ID"
}

if [ "$UNINSTALL" -eq 1 ]; then remove_owned; exit 0; fi

curl_args='--fail --silent --show-error --location --proto =https --tlsv1.2'
curl_config=$(mktemp "${TMPDIR:-/tmp}/ragtime-curl.XXXXXX")
trap 'rm -f "$curl_config"' EXIT HUP INT TERM
umask 077
printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" > "$curl_config"
[ -z "$CA_FILE" ] || printf 'cacert = "%s"\n' "$CA_FILE" >> "$curl_config"
request() {
  # shellcheck disable=SC2086
  curl $curl_args --config "$curl_config" "$@"
}
json_field() {
  # The material endpoint emits a flat, server-controlled object. Reject rather than guess.
  value=$(printf '%s' "$1" | sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -n 1)
  [ -n "$value" ] || die "Enrollment material omitted $2"
  printf '%s' "$value"
}
json_optional_field() {
  printf '%s' "$1" | sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -n 1
}
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else die 'No SHA-256 utility is available to verify the tunnel client artifact.'; fi
}
install_tunnel_service() {
  credential=$(json_field "$response" credential)
  ws_url=$(json_optional_field "$response" ws_url)
  [ -n "$ws_url" ] || ws_url=$(json_field "$material" ws_url)
  config="$STATE_DIR/$PAIRING_ID/tunnel.json"
  printf '{"server_url":"%s","pairing_id":"%s","credential":"%s","sshd_port":%s%s}' \
    "$(json_escape "$ws_url")" "$(json_escape "$PAIRING_ID")" "$(json_escape "$credential")" "$SSHD_PORT" \
    "$( [ -n "$CA_FILE" ] && printf ',"ca_file":"%s"' "$(json_escape "$CA_FILE")" )" > "$config"
  chmod 600 "$config"
  if [ "$(id -u)" -eq 0 ]; then chown "$TARGET_USER" "$config"; fi
  client=${TUNNEL_CLIENT:-$(json_optional_field "$material" tunnel_client_url)}
  [ -n "$client" ] || client="$SERVER_URL/indexes/remote/tunnel-client.py"
  case "$client" in
    https://*)
      expected=$(json_field "$material" tunnel_client_sha256)
      artifact="$STATE_DIR/$PAIRING_ID/tunnel-client"
      # shellcheck disable=SC2086
      curl $curl_args --config "$curl_config" "$client" -o "$artifact"
      [ "$(sha256_file "$artifact")" = "$expected" ] || { rm -f "$artifact"; die 'Tunnel client SHA-256 verification failed.'; }
      chmod 700 "$artifact"; [ "$(id -u)" -ne 0 ] || chown "$TARGET_USER" "$artifact"; client=$artifact;;
    *) [ -x "$client" ] || die "Tunnel client is not executable: $client";;
  esac
  case "$client" in *[!A-Za-z0-9_./-]*|'') die 'Tunnel client path contains unsupported characters.';; esac
  if command -v systemctl >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
    unit=/etc/systemd/system/ragtime-remote-"$PAIRING_ID".service
    cat > "$unit" <<EOF
[Unit]
Description=Ragtime Remote tunnel $PAIRING_ID
[Service]
User=$TARGET_USER
ExecStart=$client $config
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload && systemctl enable --now "$(basename "$unit")"
  elif command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    unit="$TARGET_HOME/.config/systemd/user/ragtime-remote-$PAIRING_ID.service"
    mkdir -p "$(dirname "$unit")"
    cat > "$unit" <<EOF
[Unit]
Description=Ragtime Remote tunnel $PAIRING_ID
[Service]
ExecStart=$client $config
Restart=always
RestartSec=5
[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload && systemctl --user enable --now "$(basename "$unit")"
    printf '%s\n' 'Tunnel persistence is limited to this logged-in user session; boot persistence requires an administrator to enable user lingering.' >&2
  elif command -v rc-service >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
    unit="/etc/init.d/ragtime-remote-$PAIRING_ID"
    log="/var/log/ragtime-remote-$PAIRING_ID.log"
    cat > "$unit" <<EOF
#!/sbin/openrc-run
name="Ragtime Remote tunnel $PAIRING_ID"
command="$client"
command_args="$config"
command_user="$TARGET_USER"
supervisor=supervise-daemon
respawn_delay=5
output_log="$log"
error_log="$log"
EOF
    chmod 700 "$unit"
    : > "$log"; chown "$TARGET_USER" "$log"; chmod 600 "$log"
    rc-update add "$(basename "$unit")" default && rc-service "$(basename "$unit")" start
  elif [ "$(uname)" = Darwin ]; then
    plist="$TARGET_HOME/Library/LaunchAgents/ragtime-remote-$PAIRING_ID.plist"
    mkdir -p "$(dirname "$plist")"
    cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd"><plist version="1.0"><dict><key>Label</key><string>ragtime-remote-$PAIRING_ID</string><key>ProgramArguments</key><array><string>$client</string><string>$config</string></array><key>RunAtLoad</key><true/><key>KeepAlive</key><true/></dict></plist>
EOF
    launchctl bootstrap "gui/$(id -u "$TARGET_USER")" "$plist"
  else
    die 'No supported persistent service manager for a non-root tunnel install; run the verified client manually with --config.'
  fi
}

preflight_tunnel_prerequisites() {
  TUNNEL_PREREQUISITE_ERROR=
  if [ -z "$TUNNEL_CLIENT" ] && ! command -v python3 >/dev/null 2>&1; then
    TUNNEL_PREREQUISITE_ERROR='Bundled tunnel mode requires Python 3.8 or newer; provide --tunnel-client or use direct SSH.'
    return 1
  fi
  if [ -z "$TUNNEL_CLIENT" ] && ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 8))' 2>/dev/null; then
    TUNNEL_PREREQUISITE_ERROR='Bundled tunnel mode requires Python 3.8 or newer; provide --tunnel-client or use direct SSH.'
    return 1
  fi
  if [ "$(uname)" = Darwin ]; then
    command -v launchctl >/dev/null 2>&1 || { TUNNEL_PREREQUISITE_ERROR='Tunnel mode requires launchctl for persistent macOS service installation.'; return 1; }
    return
  fi
  if command -v systemctl >/dev/null 2>&1; then
    [ "$(id -u)" -eq 0 ] && return
    systemctl --user show-environment >/dev/null 2>&1 && return
    TUNNEL_PREREQUISITE_ERROR='Tunnel mode needs root systemd service installation or an active systemd user session; direct SSH remains available.'
    return 1
  fi
  if command -v rc-service >/dev/null 2>&1; then
    [ "$(id -u)" -eq 0 ] && return
    TUNNEL_PREREQUISITE_ERROR='Tunnel mode on OpenRC requires root to install its supervised service; direct SSH remains available.'
    return 1
  fi
  TUNNEL_PREREQUISITE_ERROR='No supported persistent service manager is available for tunnel mode; direct SSH remains available.'
  return 1
}

if [ "$MODE" = offline ]; then
  PAIRING_ID=$OFFLINE_PAIRING_ID
  PUBLIC_KEY=$OFFLINE_PUBLIC_KEY
else
  material=$(request "$SERVER_URL/indexes/remote/enroll/material") || die 'Could not retrieve enrollment material'
  PAIRING_ID=$(json_field "$material" pairing_id)
  PUBLIC_KEY=$(json_field "$material" public_key)
fi
case "$PAIRING_ID" in *[!A-Za-z0-9_-]*|'') die 'Enrollment material returned an unsafe pairing ID';; esac
case "$PUBLIC_KEY" in 'ssh-'*' '*) ;; *) die 'Enrollment material has an invalid public key';; esac

if [ "$(id -u)" -eq 0 ] && [ "$TARGET_USER" != "$(id -un)" ]; then
  sudo -u "$TARGET_USER" mkdir -p "$SSH_DIR" "$STATE_DIR/$PAIRING_ID"
else
  mkdir -p "$SSH_DIR" "$STATE_DIR/$PAIRING_ID"
fi
chmod 700 "$SSH_DIR" "$STATE_DIR" "$STATE_DIR/$PAIRING_ID"
  touch "$KEY_FILE"; chmod 600 "$KEY_FILE"
  if [ "$(id -u)" -eq 0 ] && [ "$TARGET_USER" != "$(id -un)" ]; then chown "$TARGET_USER" "$SSH_DIR" "$KEY_FILE" "$STATE_DIR" "$STATE_DIR/$PAIRING_ID"; fi
key_tmp=$(mktemp "${TMPDIR:-/tmp}/ragtime-key.XXXXXX")
awk -v tag="${TAG_PREFIX}$PAIRING_ID" -v key="$PUBLIC_KEY" '$0 ~ tag { if (!done++) print key " " tag; next } {print} END {if (!done) print key " " tag}' "$KEY_FILE" > "$key_tmp"
cat "$key_tmp" > "$KEY_FILE"; rm -f "$key_tmp"
command -v restorecon >/dev/null 2>&1 && restorecon "$SSH_DIR" "$KEY_FILE" 2>/dev/null || true

detect_port() {
  [ -n "$SSHD_PORT" ] && { valid_port "$SSHD_PORT" || die 'Invalid --port'; printf '%s' "$SSHD_PORT"; return; }
  sshd_path=$(command -v sshd 2>/dev/null || printf /usr/sbin/sshd)
  if [ -x "$sshd_path" ]; then
    port=$($sshd_path -T 2>/dev/null | awk '$1 == "port" { print $2; exit }')
    valid_port "${port:-}" && { printf '%s' "$port"; return; }
  fi
  printf '22'
}
SSHD_PORT=$(detect_port)

ensure_sshd() {
  if [ "$(uname)" = Darwin ]; then
    if ! systemsetup -getremotelogin 2>/dev/null | grep -qi 'on'; then
      [ "$(id -u)" -eq 0 ] || die 'Remote Login is disabled. Re-run with sudo to enable it.'
      systemsetup -setremotelogin on
    fi
    return 0
  fi
  command -v sshd >/dev/null 2>&1 && return 0
  [ -x /usr/sbin/sshd ] && return 0
  [ "$(id -u)" -eq 0 ] || die 'OpenSSH server is absent. Re-run with sudo or install/enable it using your OS policy.'
  if command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y openssh-server
  elif command -v dnf >/dev/null 2>&1; then dnf install -y openssh-server
  elif command -v yum >/dev/null 2>&1; then yum install -y openssh-server
  elif command -v apk >/dev/null 2>&1; then apk add openssh
  elif command -v pacman >/dev/null 2>&1; then pacman --noconfirm -S openssh
  elif command -v zypper >/dev/null 2>&1; then zypper --non-interactive install openssh
  else die 'Unsupported package manager; install OpenSSH server manually.'; fi
  if command -v systemctl >/dev/null 2>&1; then systemctl enable --now sshd 2>/dev/null || systemctl enable --now ssh
  elif command -v rc-service >/dev/null 2>&1; then rc-update add sshd default && rc-service sshd start
  else die 'OpenSSH installed but no supported service manager was found; start sshd manually.'; fi
}

ensure_sshd
EFFECTIVE_MODE=$MODE
case "$MODE" in
  tunnel) preflight_tunnel_prerequisites || die "$TUNNEL_PREREQUISITE_ERROR" ;;
  auto)
    if ! preflight_tunnel_prerequisites; then
      EFFECTIVE_MODE=direct
      printf 'Auto mode will enroll direct SSH only: %s\n' "$TUNNEL_PREREQUISITE_ERROR" >&2
    fi
    ;;
esac
hostname_value=$(hostname 2>/dev/null || printf unknown)
addresses=$( (command -v ip >/dev/null 2>&1 && ip -o addr show 2>/dev/null | awk '{sub(/\/.*/, "", $4); print $4}') || (command -v ifconfig >/dev/null 2>&1 && ifconfig 2>/dev/null | awk '/inet6? / {print $2}' | sed 's/%.*//') || true )
addresses_json=$(printf '%s\n' "$addresses" | awk '/^[0-9a-fA-F:.]+$/ && $0 !~ /^(127\.|169\.254\.|::1$|fe80:)/ { printf "%s\"%s\"", (n++ ? "," : ""), $0; if (n == 8) exit }')
HOST_KEY_DIR=${RAGTIME_SSH_HOST_KEY_DIR:-/etc/ssh}
host_keys=$(for file in "$HOST_KEY_DIR"/ssh_host_*_key.pub; do [ -r "$file" ] && cat "$file"; done | head -n 6)
host_keys_json=$(printf '%s\n' "$host_keys" | awk 'NF >= 2 && $1 ~ /^ssh-/ { printf "%s\"%s %s\"", (n++ ? "," : ""), $1, $2 }')
[ -n "$host_keys_json" ] || die 'No SSH host keys were found; start/enable sshd and rerun before enrollment.'
python_available=false; command -v python3 >/dev/null 2>&1 && python_available=true
tunnel_errors=
[ -z "$TUNNEL_PREREQUISITE_ERROR" ] || tunnel_errors="\"$(json_escape "$TUNNEL_PREREQUISITE_ERROR")\""
facts=$(printf '{"os":"%s","username":"%s","hostname":"%s","addresses":[%s],"sshd_port":%s,"host_keys":[%s],"shared_admin_keyfile":false,"errors":[%s],"python_available":%s}' \
  "$(json_escape "$(uname | tr '[:upper:]' '[:lower:]' | sed 's/darwin/macos/; s/linux/linux/')")" "$(json_escape "$TARGET_USER")" "$(json_escape "$hostname_value")" "$addresses_json" "$SSHD_PORT" "$host_keys_json" "$tunnel_errors" "$python_available")

if [ "$MODE" = offline ]; then
  nonce_file="$STATE_DIR/$PAIRING_ID/nonce"
  if [ -r "$nonce_file" ]; then nonce=$(cat "$nonce_file"); else nonce=$( (umask 077; dd if=/dev/urandom bs=18 count=1 2>/dev/null | base64 | tr -d '=+/\n' | cut -c1-24) ); printf '%s' "$nonce" > "$nonce_file"; fi
  blob=$(printf '{"pairing_id":"%s","public_key":"%s","nonce":"%s","facts":%s,"mode":"offline"}' "$(json_escape "$PAIRING_ID")" "$(json_escape "$PUBLIC_KEY")" "$nonce" "$facts" | base64 | tr -d '\n')
  printf 'Paste this offline enrollment blob into the authenticated Ragtime UI:\n%s\n' "$blob"
  exit 0
fi

nonce_file="$STATE_DIR/$PAIRING_ID/nonce"
if [ -r "$nonce_file" ]; then nonce=$(cat "$nonce_file"); else nonce=$( (umask 077; dd if=/dev/urandom bs=18 count=1 2>/dev/null | base64 | tr -d '=+/\n' | cut -c1-24) ); printf '%s' "$nonce" > "$nonce_file"; fi
payload=$(printf '{"nonce":"%s","facts":%s,"mode":"%s"}' "$nonce" "$facts" "$EFFECTIVE_MODE")
complete_url="$SERVER_URL/indexes/remote/enroll/complete"
attempt=1
while :; do
  if response=$(request -X POST -H 'Content-Type: application/json' --data "$payload" "$complete_url"); then break; fi
  [ "$attempt" -ge 3 ] && die 'Enrollment completion failed; rerun to safely retry with the same nonce.'
  attempt=$((attempt + 1)); sleep "$attempt"
done
printf '%s\n' "$response" > "$STATE_DIR/$PAIRING_ID/complete.json"
chmod 600 "$STATE_DIR/$PAIRING_ID/complete.json"
transport=$(json_optional_field "$response" transport)
status=$(json_optional_field "$response" status)
  case "$status:$transport" in failed:*|revoked:*|expired:*|consumed:direct|consumed:'') die "Enrollment did not pair the host (status: ${status:-unknown}).";; esac
[ "$transport" != tunnel ] || install_tunnel_service
printf 'Paired %s for account %s on SSH port %s. Existing SSH authentication policies were not changed.\n' "$PAIRING_ID" "$TARGET_USER" "$SSHD_PORT"
