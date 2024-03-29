output "aws_iam_role" {
  value = aws_iam_role.role.arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "region" {
  description = "AWS region"
  value       = var.region
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = module.eks.cluster_name
} 

output "registry_repository_url" {
  description = "ECR Registry URL"
  value       = aws_ecr_repository.repo.repository_url 
}