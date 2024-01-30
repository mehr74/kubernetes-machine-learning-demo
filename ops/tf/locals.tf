locals {
  repository_name = "${var.github_owner}/${var.github_repo}"
  cluster_name = "demo-eks-${random_string.suffix.result}"
}