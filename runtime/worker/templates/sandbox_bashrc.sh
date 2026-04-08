__sandbox_update_ps1() {
  local current="${PWD:-/}"

  if [[ "$current" == "/" ]]; then
    PS1='\[\e[1;32m\]sandbox\[\e[0m\]:\[\e[1;34m\]/\[\e[0m\]$ '
  elif [[ "$current" == "/workspace" ]]; then
    PS1='\[\e[1;32m\]sandbox\[\e[0m\]:\[\e[1;34m\]$workspace\[\e[0m\]$ '
  elif [[ "$current" == /workspace/* ]]; then
    local display="${current#/workspace/}"
    PS1='\[\e[1;32m\]sandbox\[\e[0m\]:\[\e[1;34m\]$workspace/'"$display"'\[\e[0m\]$ '
  else
    PS1='\[\e[1;32m\]sandbox\[\e[0m\]:\[\e[1;34m\]'"$current"'\[\e[0m\]$ '
  fi
}
shopt -u promptvars
PROMPT_COMMAND=__sandbox_update_ps1
