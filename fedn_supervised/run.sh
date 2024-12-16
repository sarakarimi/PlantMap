# Loop from 1 to 6 and execute the script with the argument
for i in {1..6}; do
    sbatch exp.sh "$i"
done