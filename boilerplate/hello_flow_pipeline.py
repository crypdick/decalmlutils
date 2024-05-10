"""
This pipeline is boilerplate for a metaflow pipeline.

The HelloFlow has been run already, so you can use it to generate arbitrary artifacts in our Metaflow artifact store
without polluting the Flow namespace.

This is useful for doing flow surgery, such as injecting arbitrary inputs into the InferenceFlow pipeline.
"""

from metaflow import FlowSpec, Parameter, step

from mltoolkit.io.aws.cloudwatch_metrics import alert_job_finished


class HelloFlowPipeline(FlowSpec):
    srcrun_selection2hello = Parameter("srcrun-selection2hello")

    @step
    def start(self):
        import time

        self.start_time = time.time()
        self.mode = "prod"

        self.model_version = "v2023.07.06"
        self.model_code_name = "Lovely Lycaon"

        self.next(self.end)

    @step
    def end(self):
        alert_job_finished(
            mode=self.mode,
            start_time=self.start_time,
            msg="Hello Flow Pipeline finished.",
        )


if __name__ == "__main__":
    HelloFlowPipeline()
