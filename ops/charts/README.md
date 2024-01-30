# Helm Charts


After installing the infrastructure from `/ops/tf`, and building the image from 
`/src/sklearn`, you can deploy the application to the cluster using Helm. 
Use the following command to install the application:

```bash
helm upgrade --install sklearn  \
--set image.repository=$(terraform -chdir=../tf output -raw registry_repository_url) .
```