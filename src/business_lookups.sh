#! /bin/sh

# run script with list of arguments for the cutoff threshold
# ./business_lookups.sh 10 12 15 20 25

for num
do 
    echo "Setting threshold to: ${num}"
    python edit_data.py -dpcf business --id_to_indx="${num}_bus_to_indx.p" --indx_to_id="${num}_indx_to_bus.p" business_id review_count "( (review_count >= ${num}) and ( (Restaurants in categories) or (Food in categories) or (Bars in categories) ))"
done