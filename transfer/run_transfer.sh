set -e; O=${LANGDECODER_OUT:-outputs}/transfer; M=${LANGDECODER_OUT:-outputs}/models; T9=${LANGDECODER_DATA:-data}/transcripts; MASK=${MASK:-lexicon}
mkdir -p $O/text
others () { echo "lppc:cn lppc:en lppc:fr" | tr ' ' '\n' | grep -v ":$1" | paste -sd,; }
for T in cn en fr; do
  python -m langdecoder.train pretrain --sources $(others $T) --stories 1-9 --val-story story09 --epochs 8 --max-subjects 999 --init $M/trunk_smn.pt --out $O/pre_lppc_xl_$T.pt
done
for S in en zh; do
  python -m langdecoder.train pretrain --sources chen:$S --stories 1-10 --val-story story10 --epochs 8 --max-subjects 999 --init $M/trunk_smn.pt --out $O/pre_chen_$S.pt
done
for p in 30 50 70; do
  k=$((p==30?2:(p==50?4:6)))
  for T in cn en fr; do
    python -m langdecoder.train finetune --dataset lppc --lang $T --fit 1-$k --val 8 --test 9 --init $M/trunk_smn.pt    --out $O/lppc_${T}_p${p}_base.pt
    python -m langdecoder.train finetune --dataset lppc --lang $T --fit 1-$k --val 8 --test 9 --init $O/pre_lppc_xl_$T.pt --out $O/lppc_${T}_p${p}_xl.pt
  done
  k=$((p==30?3:(p==50?4:6)))
  for T in en zh; do S=$([ $T = en ] && echo zh || echo en)
    python -m langdecoder.train finetune --dataset chen --lang $T --fit 1-$k --val 9 --test 10 --init $M/trunk_smn.pt  --out $O/chen_${T}_p${p}_base.pt
    python -m langdecoder.train finetune --dataset chen --lang $T --fit 1-$k --val 9 --test 10 --init $O/pre_chen_$S.pt --out $O/chen_${T}_p${p}_xl.pt
  done
done
for ck in $O/lppc_*_p*_*.pt $O/chen_*_p*_*.pt; do n=$(basename $ck .pt); D=${n%%_*}; L=$(echo $n | cut -d_ -f2)
  ST=$([ $D = lppc ] && echo story09 || echo story10); G="$T9/$D/$L/story0[1-8].TextGrid"; [ $D = chen ] && G="$T9/$D/$L/story0[1-9].TextGrid"
  python persub_beam.py --decoder $ck --dataset $D --lang $L --story $ST --train-grids "$G" --mask $MASK --out-dir $O/text --name $n
  for g in $O/text/gen_${n}_sub-*.json; do s=${g#$O/text/gen_}
    python -m langdecoder.score --generation $g --transcript $T9/$D/$L/$ST.TextGrid --language $([ $L = cn ] && echo zh || echo $L) --out $O/text/score_$s; done
done
for T in cn en fr; do python occlusion_30pct.py --decoder $O/lppc_${T}_p30_xl.pt --lang $T --out $O/contrib_lppc_${T}_p30_xl.npz; done
