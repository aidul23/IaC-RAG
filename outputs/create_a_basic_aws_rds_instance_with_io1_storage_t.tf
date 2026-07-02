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


resource "aws_subnet" "primary" {
  # Subnet created for the RDS instance — not using an existing one
  cidr_block = "10.0.1.0/24"
}

resource "aws_security_group" "main" {
  # Security group created for the RDS instance — not using an existing one
  description = "RDS security group"
  default_language = "en"

  tags = merge(var.tags, { Name = "rds-security-group" })
}

resource "aws_db_instance" "primary" {
  # RDS instance created by the request — not using an existing one
  db_instance_class = "io1"
  vpc_security_group_ids = [var.security_group_id]
  allocated_storage    = 20

  tags = merge(var.tags, { Name = "rds-instance" })
}

resource "aws_security_group_rule" "main" {
  # Security rule created for the RDS instance — not using an existing one
  description = "RDS security rule"
  cidr_blocks = ["0.0.0.0/0"]
  type        = "ingress"

  depends_on  = [aws_security_group.main]
  source_security_group_id = var.security_group_id
}