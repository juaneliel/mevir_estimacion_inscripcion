FROM nginx:stable-alpine
FROM python:3.10-slim 



WORKDIR /usr/share/nginx/html/mevir-estimacion-inscripcion
COPY . .


# RUN /usr/local/bin/python -m pip install --upgrade pip

# RUN pip install requests beautifulsoup4 python-dotenv
RUN pip install --no-cache-dir -r /usr/share/nginx/html/mevir-estimacion-inscripcion/App/requirements.txt 






# #Agrego bash a la imagen y ejecuto env.sh
RUN chmod +x start.sh

EXPOSE 80

# #Inicio el servidor NGINX
CMD ["/bin/bash", "-c", "/usr/share/nginx/html/mevir-estimacion-inscripcion/start.sh && nginx -g \"daemon off;\""]




