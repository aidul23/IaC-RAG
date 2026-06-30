"""
Keyword-to-resource-type taxonomy for Terraform AWS provider.

Maps natural-language service/feature terms that appear in IaC prompts
to the specific aws_* resource types that need to be generated.

Used by rag_retriever to guarantee the right Argument Reference sections
are always included, regardless of their semantic similarity score.
"""

from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Taxonomy table
#
# Keys   : lowercase keyword phrases, longest first within a group so that
#          "application load balancer" matches before "load balancer".
# Values : ordered list of primary resource types to include.
#          Put the most important type first; retriever uses this order.
# ---------------------------------------------------------------------------
_TAXONOMY: dict[str, list[str]] = {

    # ── Compute ──────────────────────────────────────────────────────────
    "ec2 instance":         ["aws_instance", "aws_security_group"],
    "ec2":                  ["aws_instance", "aws_security_group"],
    "instance":             ["aws_instance"],

    "lambda function":      ["aws_lambda_function", "aws_iam_role", "aws_cloudwatch_log_group"],
    "lambda":               ["aws_lambda_function", "aws_iam_role"],

    "ecs fargate":          ["aws_ecs_service", "aws_ecs_task_definition",
                             "aws_ecs_cluster", "aws_iam_role"],
    "fargate":              ["aws_ecs_service", "aws_ecs_task_definition",
                             "aws_ecs_cluster", "aws_iam_role"],
    "ecs service":          ["aws_ecs_service", "aws_ecs_task_definition", "aws_ecs_cluster"],
    "ecs task":             ["aws_ecs_task_definition", "aws_iam_role"],
    "ecs cluster":          ["aws_ecs_cluster"],
    "ecs":                  ["aws_ecs_service", "aws_ecs_task_definition", "aws_ecs_cluster"],

    "eks cluster":          ["aws_eks_cluster", "aws_eks_node_group", "aws_iam_role"],
    "eks node group":       ["aws_eks_node_group", "aws_iam_role"],
    "eks":                  ["aws_eks_cluster", "aws_eks_node_group"],

    "elastic beanstalk":    ["aws_elastic_beanstalk_application",
                             "aws_elastic_beanstalk_environment"],
    "beanstalk":            ["aws_elastic_beanstalk_application",
                             "aws_elastic_beanstalk_environment"],

    "auto scaling group":   ["aws_autoscaling_group", "aws_launch_template"],
    "autoscaling group":    ["aws_autoscaling_group", "aws_launch_template"],
    "launch template":      ["aws_launch_template"],
    "auto scaling":         ["aws_autoscaling_group", "aws_launch_template"],

    "application autoscaling": ["aws_appautoscaling_target", "aws_appautoscaling_policy"],
    "app autoscaling":      ["aws_appautoscaling_target", "aws_appautoscaling_policy"],

    # ── Storage ───────────────────────────────────────────────────────────
    "s3 bucket":            ["aws_s3_bucket",
                             "aws_s3_bucket_versioning",
                             "aws_s3_bucket_server_side_encryption_configuration",
                             "aws_s3_bucket_policy",
                             "aws_s3_bucket_public_access_block"],
    "s3":                   ["aws_s3_bucket",
                             "aws_s3_bucket_versioning",
                             "aws_s3_bucket_server_side_encryption_configuration"],

    "ebs volume":           ["aws_ebs_volume", "aws_volume_attachment"],
    "ebs":                  ["aws_ebs_volume"],

    "efs file system":      ["aws_efs_file_system", "aws_efs_mount_target",
                             "aws_efs_access_point"],
    "elastic file system":  ["aws_efs_file_system", "aws_efs_mount_target"],
    "efs":                  ["aws_efs_file_system", "aws_efs_mount_target"],

    # ── Database ─────────────────────────────────────────────────────────
    "rds instance":         ["aws_db_instance", "aws_db_subnet_group",
                             "aws_db_parameter_group", "aws_security_group"],
    "rds cluster":          ["aws_rds_cluster", "aws_rds_cluster_instance",
                             "aws_db_subnet_group"],
    "aurora cluster":       ["aws_rds_cluster", "aws_rds_cluster_instance",
                             "aws_db_subnet_group"],
    "aurora":               ["aws_rds_cluster", "aws_rds_cluster_instance"],
    "rds":                  ["aws_db_instance", "aws_db_subnet_group"],

    "dynamodb table":       ["aws_dynamodb_table"],
    "dynamodb":             ["aws_dynamodb_table"],

    "elasticache cluster":  ["aws_elasticache_cluster", "aws_elasticache_subnet_group"],
    "elasticache":          ["aws_elasticache_cluster", "aws_elasticache_subnet_group"],
    "redis":                ["aws_elasticache_replication_group",
                             "aws_elasticache_subnet_group"],
    "memcached":            ["aws_elasticache_cluster", "aws_elasticache_subnet_group"],

    "redshift cluster":     ["aws_redshift_cluster", "aws_redshift_subnet_group"],
    "redshift":             ["aws_redshift_cluster", "aws_redshift_subnet_group"],

    # ── Networking ────────────────────────────────────────────────────────
    "vpc":                  ["aws_vpc", "aws_subnet",
                             "aws_internet_gateway", "aws_route_table",
                             "aws_route_table_association"],
    "subnet":               ["aws_subnet", "aws_vpc"],
    "internet gateway":     ["aws_internet_gateway", "aws_vpc"],
    "nat gateway":          ["aws_nat_gateway", "aws_eip"],
    "route table":          ["aws_route_table", "aws_route_table_association"],
    "security group":       ["aws_security_group", "aws_vpc_security_group_ingress_rule",
                             "aws_vpc_security_group_egress_rule"],
    "vpc peering":          ["aws_vpc_peering_connection",
                             "aws_vpc_peering_connection_accepter"],
    "transit gateway":      ["aws_ec2_transit_gateway",
                             "aws_ec2_transit_gateway_vpc_attachment"],

    "application load balancer": ["aws_lb", "aws_lb_listener", "aws_lb_target_group"],
    "network load balancer":     ["aws_lb", "aws_lb_listener", "aws_lb_target_group"],
    "alb":                  ["aws_lb", "aws_lb_listener", "aws_lb_target_group"],
    "nlb":                  ["aws_lb", "aws_lb_listener", "aws_lb_target_group"],
    "load balancer":        ["aws_lb", "aws_lb_listener", "aws_lb_target_group"],
    "target group":         ["aws_lb_target_group"],
    "listener":             ["aws_lb_listener"],

    "route53 hosted zone":  ["aws_route53_zone", "aws_route53_record"],
    "route53 zone":         ["aws_route53_zone", "aws_route53_record"],
    "hosted zone":          ["aws_route53_zone", "aws_route53_record"],
    "route53 record":       ["aws_route53_record"],
    "query logging":        ["aws_route53_query_log", "aws_cloudwatch_log_group",
                             "aws_cloudwatch_log_resource_policy"],
    "query log":            ["aws_route53_query_log", "aws_cloudwatch_log_group",
                             "aws_cloudwatch_log_resource_policy"],
    "route53":              ["aws_route53_zone", "aws_route53_record"],
    "dns":                  ["aws_route53_zone", "aws_route53_record"],

    "cloudfront":           ["aws_cloudfront_distribution", "aws_cloudfront_origin_access_identity"],

    "api gateway v2":       ["aws_apigatewayv2_api", "aws_apigatewayv2_stage",
                             "aws_apigatewayv2_integration"],
    "http api":             ["aws_apigatewayv2_api", "aws_apigatewayv2_stage"],
    "websocket api":        ["aws_apigatewayv2_api", "aws_apigatewayv2_stage"],
    "api gateway":          ["aws_api_gateway_rest_api", "aws_api_gateway_resource",
                             "aws_api_gateway_method", "aws_api_gateway_integration",
                             "aws_api_gateway_deployment", "aws_api_gateway_stage"],

    # ── IAM ───────────────────────────────────────────────────────────────
    "iam role":             ["aws_iam_role", "aws_iam_role_policy_attachment",
                             "data.aws_iam_policy_document"],
    "iam policy":           ["aws_iam_policy", "data.aws_iam_policy_document"],
    "iam user":             ["aws_iam_user", "aws_iam_user_policy",
                             "aws_iam_access_key"],
    "iam group":            ["aws_iam_group", "aws_iam_group_membership",
                             "aws_iam_group_policy_attachment"],
    "iam instance profile": ["aws_iam_instance_profile", "aws_iam_role"],
    "policy document":      ["data.aws_iam_policy_document"],
    # No catch-all "iam" entry — it fires on "IAM" in almost every AWS prompt
    # and crowds out the services that were explicitly named.

    # ── Messaging / Events ────────────────────────────────────────────────
    "sns topic":            ["aws_sns_topic", "aws_sns_topic_subscription",
                             "aws_sns_topic_policy"],
    "sns":                  ["aws_sns_topic", "aws_sns_topic_subscription"],

    "sqs queue":            ["aws_sqs_queue", "aws_sqs_queue_policy"],
    "sqs":                  ["aws_sqs_queue", "aws_sqs_queue_policy"],

    "kinesis stream":       ["aws_kinesis_stream"],
    "kinesis firehose":     ["aws_kinesis_firehose_delivery_stream"],
    "kinesis":              ["aws_kinesis_stream"],

    "eventbridge rule":     ["aws_cloudwatch_event_rule", "aws_cloudwatch_event_target"],
    "eventbridge":          ["aws_cloudwatch_event_rule", "aws_cloudwatch_event_target",
                             "aws_cloudwatch_event_bus"],
    "cloudwatch events":    ["aws_cloudwatch_event_rule", "aws_cloudwatch_event_target"],

    "step functions":       ["aws_sfn_state_machine", "aws_iam_role"],
    "state machine":        ["aws_sfn_state_machine", "aws_iam_role"],

    # ── Monitoring / Logging ──────────────────────────────────────────────
    "cloudwatch log group": ["aws_cloudwatch_log_group"],
    "log group":            ["aws_cloudwatch_log_group"],
    "cloudwatch alarm":     ["aws_cloudwatch_metric_alarm"],
    "metric alarm":         ["aws_cloudwatch_metric_alarm"],
    "cloudwatch dashboard": ["aws_cloudwatch_dashboard"],
    "cloudwatch":           ["aws_cloudwatch_log_group"],

    # ── Security ─────────────────────────────────────────────────────────
    "kms key":              ["aws_kms_key", "aws_kms_alias"],
    "kms":                  ["aws_kms_key", "aws_kms_alias"],

    "acm certificate":      ["aws_acm_certificate", "aws_acm_certificate_validation",
                             "data.aws_acm_certificate"],
    "ssl certificate":      ["aws_acm_certificate", "aws_acm_certificate_validation"],
    "acm":                  ["aws_acm_certificate", "aws_acm_certificate_validation"],

    "secrets manager":      ["aws_secretsmanager_secret",
                             "aws_secretsmanager_secret_version"],
    "secret":               ["aws_secretsmanager_secret",
                             "aws_secretsmanager_secret_version"],

    "waf":                  ["aws_wafv2_web_acl", "aws_wafv2_web_acl_association"],
    "guardduty":            ["aws_guardduty_detector"],
    "config rule":          ["aws_config_config_rule", "aws_config_configuration_recorder"],
    "access analyzer":      ["aws_accessanalyzer_analyzer"],

    # ── CI/CD / Build ─────────────────────────────────────────────────────
    "codepipeline":         ["aws_codepipeline", "aws_s3_bucket", "aws_iam_role"],
    "codebuild":            ["aws_codebuild_project", "aws_iam_role",
                             "aws_cloudwatch_log_group"],
    "codecommit":           ["aws_codecommit_repository"],
    "codedeploy":           ["aws_codedeploy_app", "aws_codedeploy_deployment_group"],

    # ── Other common services ─────────────────────────────────────────────
    "ecr repository":       ["aws_ecr_repository", "aws_ecr_lifecycle_policy"],
    "ecr":                  ["aws_ecr_repository"],

    "opensearch":           ["aws_opensearch_domain"],
    "elasticsearch":        ["aws_opensearch_domain"],

    "glue":                 ["aws_glue_catalog_database", "aws_glue_crawler",
                             "aws_glue_job"],
    "athena":               ["aws_athena_workgroup", "aws_athena_database"],
    "emr":                  ["aws_emr_cluster"],
    "msk":                  ["aws_msk_cluster", "aws_msk_configuration"],
    "kafka":                ["aws_msk_cluster"],

    "ses":                  ["aws_ses_domain_identity", "aws_ses_email_identity"],
    "cognito":              ["aws_cognito_user_pool",
                             "aws_cognito_user_pool_client"],
    "amplify":              ["aws_amplify_app", "aws_amplify_branch"],
    "apprunner":            ["aws_apprunner_service"],

    "backup":               ["aws_backup_plan", "aws_backup_vault", "aws_backup_selection"],
    "systems manager":      ["aws_ssm_parameter", "aws_ssm_document"],
    "ssm parameter":        ["aws_ssm_parameter"],
    "parameter store":      ["aws_ssm_parameter"],
}

# Pre-compile patterns sorted longest-first so multi-word phrases match before
# their constituent single words
_PATTERNS: list[tuple[re.Pattern, list[str]]] = sorted(
    [
        (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), rts)
        for kw, rts in _TAXONOMY.items()
    ],
    key=lambda pair: -len(pair[0].pattern),  # longest pattern first
)


def map_prompt_to_resources(prompt: str, max_per_match: int = 2) -> list[str]:
    """
    Scan the prompt for known service/feature keywords and return an ordered,
    deduplicated list of aws_* resource types that should be retrieved.

    max_per_match caps how many types any one keyword match contributes so that
    a single service (e.g. S3 with 5 sub-resources) can't fill all k slots and
    push other mentioned services (e.g. CloudFront, Route53) out of the results.

    The list is ordered by first keyword match in the prompt, so resources from
    earlier mentions come first.
    """
    seen: set[str] = set()
    result: list[str] = []

    for pattern, resource_types in _PATTERNS:
        if pattern.search(prompt):
            added = 0
            for rt in resource_types:
                if rt not in seen:
                    seen.add(rt)
                    result.append(rt)
                    added += 1
                    if added >= max_per_match:
                        break

    return result
