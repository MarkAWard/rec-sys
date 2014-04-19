#! /bin/sh

# run script with list of arguments for the cutoff threshold                                                                                                           
# ./user_lookups.sh 5 8 10 12 15 20   

for num
do 
    echo "Setting threshold to: ${num}"
    python edit_data.py -dpcf user --id_to_indx="${num}_user_to_indx.p" --indx_to_id="${num}_indx_to_user.p" user_id review_count "(review_count >= ${num})"
done