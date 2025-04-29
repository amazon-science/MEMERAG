Evidence Passages:

{% for passage in context %}
{{loop.index}}: {{passage.text}}
{% endfor %}

Answer:

{{answer_segment}}

Now provided your label directly as "Supported" or "Not Supported".