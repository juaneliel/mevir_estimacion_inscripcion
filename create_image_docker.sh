#!/bin/bash



if [ -z "$1" ]
then
   echo "Ingrese la version de la imagen"
  exit 1
fi

 
sudo docker build -t mevir_estimacion_inscripcion-test  -f Dockerfile .

sudo docker tag mevir_estimacion_inscripcion-test 192.168.100.55:5000/mevir_estimacion_inscripcion:$1

sudo docker push 192.168.100.55:5000/mevir_estimacion_inscripcion:$1