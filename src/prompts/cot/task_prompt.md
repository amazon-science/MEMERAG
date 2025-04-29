Evidence Passages:

{% for passage in context %}
{{loop.index}}: {{passage.text}}
{% endfor %}

Answer:

{{answer_segment}}

Provide your rationale and your factuality judgment ("Supported", "Not Supported")