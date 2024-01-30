# Terraform stack 

The repository contains terraform sources to build the following infrastructure:
* EKS cluster
* ECR repository


## Usage 
To build the infrastructure, run the following commands:
```bash
terraform init
```

```bash
terraform apply -var-file=terraform.tfvars
```

After the infrastructure is built, you can use the following command to get the kubeconfig file:
```bash
aws eks --region $(terraform output -raw region) update-kubeconfig \
    --name $(terraform output -raw cluster_name)
```

Use the following command to login to the ECR repository with docker:
```bash
aws ecr get-login-password --region $(terraform output -raw region) | \
  docker login -u AWS --password-stdin $(terraform output -raw registry_repository_url)
```



