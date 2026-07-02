terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type        = string
  description = "The AWS Region to create resources in."
}

variable "vpc_id" {
  type        = string
  description = "The ID of the VPC to associate with the egress-only internet gateway."
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "A map of tags to assign to the resource."
}

resource "aws_egress_only_internet_gateway" "main" {
  # Egress-only internet gateway attached to the specified VPC
  vpc_id = var.vpc_id

  # Add a Name tag to the egress-only internet gateway
  tags = merge(var.tags, {
    Name = "Egress-Only Internet Gateway"
  })
}

variable "selected_id" {
  type        = string
  description = "The ID of the pre-existing VPC referenced by the request."
}

resource "aws_vpc" "selected" {
  # Pre-existing VPC referenced by the request — not created here
}