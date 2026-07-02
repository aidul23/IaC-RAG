terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "tags" {
  description = "Additional tags applied to all resources"
  type        = map(string)
  default     = {}
}

variable "selected_id" {
  description = "ID of the existing aws_vpc"
  type        = string
}


resource "aws_egress_only_internet_gateway" "main" {
  # Create an egress-only internet gateway attached to the specified VPC
  comment = "Egress-only internet gateway attached to the specified VPC"

  vpc_id = var.selected_vpc_id
  tags = merge(var.tags, { Name = "descriptive-name" })
}