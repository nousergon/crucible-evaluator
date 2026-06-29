"""Producers — small GH-API/AWS collectors that write the S3 rollups the
report-card tiles read but the evaluator's own Lambda role cannot source itself.

The evaluator Lambda can reach S3 + AWS APIs (CloudWatch, Step Functions) but
holds no GitHub token, so substrate signals that live in GitHub Actions history
need a producer running where a token IS available. ``deploy_success`` is the
first such producer — invoked weekly from the Director (Layer C), the one
component that already authenticates to GitHub and writes to the research bucket.
"""
