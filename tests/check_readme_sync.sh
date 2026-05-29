#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tests/check_readme_sync.sh [--check|--write]

Checks or updates the README.md collapsible snippets generated from
.env.example and docker-compose.yml.
EOF
}

mode="check"

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --check)
      mode="check"
      ;;
    --write)
      mode="write"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
fi

write_output() {
  local name="$1"
  local value="$2"

  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "${name}=${value}" >> "$GITHUB_OUTPUT"
  fi
}

update_section() {
  local input_file="$1"
  local output_file="$2"
  local start_marker="$3"
  local source_file="$4"
  local indent="$5"

  if [[ ! -f "$source_file" ]]; then
    echo "Source file not found: $source_file" >&2
    exit 1
  fi

  local in_section=false
  local in_code_block=false
  local section_found=false

  : > "$output_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == *"$start_marker"* ]]; then
      in_section=true
      section_found=true
      echo "$line" >> "$output_file"
      continue
    fi

    if [[ "$in_section" == true ]]; then
      if [[ "$in_code_block" == false && "$line" =~ ^[[:space:]]*\`\`\` ]]; then
        in_code_block=true
        echo "$line" >> "$output_file"

        while IFS= read -r src_line || [[ -n "$src_line" ]]; do
          if [[ -n "$src_line" ]]; then
            echo "${indent}${src_line}" >> "$output_file"
          else
            echo "" >> "$output_file"
          fi
        done < "$source_file"
        continue
      elif [[ "$in_code_block" == true && "$line" =~ ^[[:space:]]*\`\`\`[[:space:]]*$ ]]; then
        echo "$line" >> "$output_file"
        in_code_block=false
        in_section=false
        continue
      elif [[ "$in_code_block" == true ]]; then
        continue
      fi
    fi

    echo "$line" >> "$output_file"
  done < "$input_file"

  if [[ "$section_found" == false ]]; then
    echo "Failed to find README section with marker: $start_marker" >&2
    exit 1
  fi
}

readme_file="README.md"

if [[ ! -f "$readme_file" ]]; then
  echo "README.md not found" >&2
  exit 1
fi

first_pass=$(mktemp)
expected_readme=$(mktemp)
trap 'rm -f "$first_pass" "$expected_readme"' EXIT

update_section \
  "$readme_file" \
  "$first_pass" \
  "   <summary>Click to expand .env template</summary>" \
  ".env.example" \
  "   "

update_section \
  "$first_pass" \
  "$expected_readme" \
  "   <summary>Click to expand docker-compose.yml</summary>" \
  "docker-compose.yml" \
  "   "

if cmp -s "$readme_file" "$expected_readme"; then
  write_output changed false
  echo "README.md snippets match .env.example and docker-compose.yml."
  exit 0
fi

write_output changed true

if [[ "$mode" == "write" ]]; then
  cp "$expected_readme" "$readme_file"
  echo "README.md snippets updated from .env.example and docker-compose.yml."
  exit 0
fi

echo "README.md snippets are out of sync with .env.example or docker-compose.yml." >&2
echo "Run: bash tests/check_readme_sync.sh --write" >&2
diff -u --label "README.md (current)" --label "README.md (expected)" "$readme_file" "$expected_readme" >&2 || true
exit 1