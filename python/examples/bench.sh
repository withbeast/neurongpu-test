sim="python brunel_array.py" # bsim ngpu (all sims)
sizes=({200000000..1600000000..200000000})

echo "[" > results.json

run() {
	echo $1
	eval "$1 >> results.json"
	echo -n "," >> results.json
}

export CUDA_VISIBLE_DEVICES=0

# simtime sparse
for size in ${sizes[@]}
do
	run "$sim --bench sim --gpu single --model synth --pconnect 0.00156 --pfire 0.005 --delay 1 --nsyn $size"
done

# simtime
for model in vogels brunel
do
	for size in ${sizes[@]}
	do
		run "$sim --bench sim --gpu single --model $model --nsyn $size"
	done
done

# setup time
for size in ${sizes[@]}
do
	run "$sim --bench setup --gpu single --model synth --pconnect 0.05 --pfire 0.005 --delay 1 --nsyn $size"
done

# TODO: speedup (as function of delay) (metrics, viz, ..)

echo -n "{}]" >> results.json
