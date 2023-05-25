start=`date +%s`
sleep 1.761

end=`date +%s`
duration=$((end-start))
echo $duration
duration=`date -u -d @${duration} +"%T"`
echo -e "Runtime: $duration"