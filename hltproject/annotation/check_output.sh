lineno=$(wc -l test_stage_2_with_labels.tsv | cut -d' ' -f1)
echo "checking the first $lineno lines"
diff <(cut -f1-6,8,9,11 test_stage_2_with_labels.tsv) <(head -n$lineno test_stage_2.tsv)

echo "(no diff output means everything is ok)"