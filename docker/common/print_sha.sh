#!/bin/bash

# Only execute on rank 0 when running with Slurm
if [[ -n "${SLURM_PROCID}" ]] && [[ "${SLURM_PROCID}" != "0" ]]; then
    exit 0
fi

BOLD=$'\e[1m'
GREEN=$'\e[32m'
YELLOW=$'\e[33m'
BLUE=$'\e[34m'
RESET=$'\e[0m'

repos=(
    "Megatron-Bridge,/opt/Megatron-Bridge"
    "Megatron-LM,/opt/Megatron-Bridge/3rdparty/Megatron-LM"
    "DeepEP,/opt/DeepEP"
    "Export-Deploy,/opt/Export-Deploy"
    "Run,/opt/Run"
)

OUTPUT_JSON="${1:-repo_status.json}"
echo "[]" > "$OUTPUT_JSON"

print_row() {
    # $1=Name, $2=SHA, $3=Date, $4=Title, $5=URL, $6=ColorName
    printf "%b%-18s%b  %b%-10s%b  %b%-12s%b  %-40s  %b%s%b\n" \
        "$6" "$1" "$RESET" \
        "$YELLOW" "$2" "$RESET" \
        "$BLUE" "$3" "$RESET" \
        "${4:0:40}" \
        "$CYAN" "$5" "$RESET"
}

printf "${BOLD}%-18s  %-10s  %-12s  %-40s  %s${RESET}\n" "REPO NAME" "SHA" "DATE" "COMMIT TITLE" "URL"
echo "------------------------------------------------------------------------------------------------------------------------------------"

for entry in "${repos[@]}"; do
    name="${entry%%,*}"
    path="${entry#*,}"

    j_sha=""
    j_date=""
    j_title=""
    j_url=""
    j_error=""

    if [ -d "$path" ] && git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git_log=$(git -C "$path" log -1 --format="%h|%cs|%s")
        
        IFS='|' read -r sha date title <<< "$git_log"

        remote_url=$(git -C "$path" config --get remote.origin.url)
        clean_url=${remote_url%.git}
        # Replace git@github.com: with https://github.com/
        if [[ "$clean_url" == git@* ]]; then
            clean_url="${clean_url/git@/https://}"
            clean_url="${clean_url/:/\/}"
        fi
        commit_url="${clean_url}/commit/${sha}"

        print_row "$name" "$sha" "$date" "$title" "$commit_url" "$GREEN"

        j_sha="$sha"
        j_date="$date"
        j_title="$title"
        j_url="$commit_url"
    else
        print_row "$name" "ERROR" "-" "Path not found" "-" "$GREEN"
        j_error="Path not found or not a git repo"
    fi

    tmp_json=$(mktemp)
    
    jq --arg name "$name" \
       --arg path "$path" \
       --arg sha "$j_sha" \
       --arg date "$j_date" \
       --arg title "$j_title" \
       --arg url "$j_url" \
       --arg err "$j_error" \
       '. + [{
           "repo_name": $name, 
           "path": $path, 
           "commit_sha": $sha, 
           "commit_date": $date, 
           "commit_title": $title, 
           "commit_url": $url, 
           "error": $err
       }]' "$OUTPUT_JSON" > "$tmp_json" && mv "$tmp_json" "$OUTPUT_JSON"

done

echo ""
echo -e "${BOLD}JSON report saved to: ${RESET}${OUTPUT_JSON}"
