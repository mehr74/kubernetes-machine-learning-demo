variable "region" {
  description = "AWS region to deploy to"
  type = string
}

variable "name" {
  type = string
}

variable "tags" {
  type = map(string)
}

variable "github_owner" {
  type = string
}

variable "github_repo" {
  type = string
}

variable "github_oidc_provider_arn" {
  type = string
}