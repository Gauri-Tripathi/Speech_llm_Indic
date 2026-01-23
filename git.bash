dates=("2026-01-18" "2026-01-19" "2026-01-20" "2026-01-21" "2026-01-22" "2026-01-23" "2026-01-24" "2026-01-25")

i=0
git status -s | while read status file; do
    date=${dates[$((i % ${#dates[@]}))]}
    git add "$file"
    GIT_AUTHOR_DATE="$date 12:00:00" GIT_COMMITTER_DATE="$date 12:00:00" git commit -m "update $file"
    i=$((i + 1))
done