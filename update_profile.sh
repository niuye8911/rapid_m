#!/bin/bash
# copy the cost and mv profile from rapidS

apps="facedetect swaptions bodytrack ferret svm nn x264"

for app in $apps
do
 cp ~/Research/rapidlib-linux/modelConstr/Rapids/outputs/$app/$app-mv.fact ~/Research/rapid_m_backend_server/outputs/$app/mv.csv
 cp ~/Research/rapidlib-linux/modelConstr/Rapids/outputs/$app/$app-mv.fact /var/www/html/rapid_server/storage/apps/algaesim-$app/mv.csv
 cp ~/Research/rapidlib-linux/modelConstr/Rapids/outputs/$app/$app-cost.fact ~/Research/rapid_m_backend_server/outputs/$app/cost.csv
 cp ~/Research/rapidlib-linux/modelConstr/Rapids/outputs/$app/$app-cost.fact /var/www/html/rapid_server/storage/apps/algaesim-$app/cost.csv
 cp ~/Research/rapidlib-linux/modelConstr/Rapids/outputs/$app/$app-*.csv ~/Research/rapid_m_backend_server/testData/halfandhalf
done
