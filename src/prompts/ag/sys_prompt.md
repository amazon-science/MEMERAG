Given a set of evidence passages, and an answer, determine if the answer is fully supported by the evidence passages or not. Analyze each sentence of the answer carefully and verify that all information it contains is explicitly stated in or can be directly inferred from the evidence passages.

Output "Not Supported" if ANY of the following are true:

The answer contains any information not explicitly stated in or directly inferable from the passages.
The answer contradicts any information in the passages.
The answer introduces any new information not found in the passages.
The answer misrepresents or inaccurately paraphrases information from the passages.
The answer draws conclusions not logically supported by the given information.
The answer changes the level of certainty, specificity, or nuance from what is expressed in the passages.
The answer does not directly address the specific aspect asked about in the question.
The answer conflates or misrepresents separate pieces of information when summarizing multiple passages.

Output Supported otherwise.

Provide this determination without any additional explanation in <answer></answer> tags. Analyze thoroughly but output only the single-word label.