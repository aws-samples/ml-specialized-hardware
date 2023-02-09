# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
ARG REGION_NAME=us-east-1
ARG ACCOUNT_ID=763104351884
FROM $ACCOUNT_ID.dkr.ecr.$REGION_NAME.amazonaws.com/pytorch-inference-neuron:1.10.2-neuron-py37-sdk1.19.0-ubuntu18.04
RUN echo '\
--- /opt/conda/lib/python3.7/site-packages/sagemaker_inference/transformer.py    2022-08-23 17:26:42.000000000 +0000\n\
+++ /opt/conda/lib/python3.7/site-packages/sagemaker_inference/transformer_.py   2022-12-07 13:15:09.753360938 +0000\n\
@@ -250,9 +250,9 @@\n\
                 (response_data, content_type)\n\
\n\
         """\n\
-        data = self._run_handler_function(self._input_fn, *(input_data, content_type))\n\
-        prediction = self._run_handler_function(self._predict_fn, *(data, model))\n\
-        result = self._run_handler_function(self._output_fn, *(prediction, accept))\n\
+        data = self._run_handler_function(self._input_fn, *(input_data, content_type, context))\n\
+        prediction = self._run_handler_function(self._predict_fn, *(data, model, context))\n\
+        result = self._run_handler_function(self._output_fn, *(prediction, accept, context))\n\
         return result\n\
\n\
     def _run_handler_function(self, func, *argv):' > /tmp/transformer.py.patch

RUN patch /opt/conda/lib/python3.7/site-packages/sagemaker_inference/transformer.py /tmp/transformer.py.patch
RUN wget --quiet --output-document=/tmp/libjpeg.deb https://netactuate.dl.sourceforge.net/project/libjpeg-turbo/2.1.4/libjpeg-turbo-official_2.1.4_amd64.deb && dpkg -i /tmp/libjpeg.deb && rm -f /tmp/libjpeg.deb
RUN pip3 install PyTurboJPEG
