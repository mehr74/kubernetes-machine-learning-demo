# sklearn




To build the docker image use the following command:

```bash
docker build --tag \
$(terraform -chdir=../../ops/tf output -raw registry_repository_url):latest .
```

To push the image to the ECR repository use the following command:
```bash
docker push $(terraform -chdir=../../ops/tf output -raw registry_repository_url):latest
```