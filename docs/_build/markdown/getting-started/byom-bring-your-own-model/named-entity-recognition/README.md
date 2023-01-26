# Named Entity Recognition

NER is a sequence tagging problem, where given an input document, the task is to correctly identify the span boundaries for various entities and also classify the spans into correct entity types.&#x20;

Galileo supports NER for various tagging schema including - BIO, BIOES, and BILOU. Additionally, you can use Galileo for other span classification tasks that follow similar schemas. Here's an example:

```
input = "Galileo was an Italian astronomer born in Pisa, and he discovered the moons of planet Jupiter"
output = [{"span_text": "Galileo", "start": 0, "end": 1, "label": "PERSON"},
          {"span_text": "Italian", "start": 3, "end": 4, "label": "MISCELANEOUS"},
          {"span_text": "Pisa", "start": 6, "end": 7, "label": "LOCATION"},
          {"span_text": "Jupiter", "start": 13, "end": 14, "label": "LOCATION"}]
```

### Data format requirements:

Galileo requires the following inputs for logging NER data:\
\* `ids` - A unique ID per sample (not per span). We reccomend simply using an index as the ids: `range(0, len(data))`\
\* `text` - the full sample text (`List[str]`)\
\* `gold_spans` - the list of gold spans, per sample (`List[List[Dict]]`)\
&#x20;                             The dictionary has the required keys `start` , `end`, `label`\
\* `text_token_indices` - These are the token boundaries for each text sample, with 1 list per sample. For a list of text inputs, this would be of type `List[List[Tuple[int,int]]]`\
``\
``For more details and examples regarding logging input NER data, see our API docs for one of:\
\* [`log_data_samples`](../../../python-library-api/logging-data/dq.log\_data\_samples.md) (for more manual logging)\
\* [`log_dataset`](../../../python-library-api/logging-data/dq.log\_dataset.md) (if you have your data in a pandas dataframe or huggingface dataset)

### Supported frameworks:

{% content-ref url="../../bring-your-own-model/text-classification/text-classification-pytorch.md" %}
[text-classification-pytorch.md](../../bring-your-own-model/text-classification/text-classification-pytorch.md)
{% endcontent-ref %}

{% content-ref url="../../bring-your-own-model/text-classification/text-classification-tensorflow.md" %}
[text-classification-tensorflow.md](../../bring-your-own-model/text-classification/text-classification-tensorflow.md)
{% endcontent-ref %}

{% content-ref url="named-entity-recognition-spacy.md" %}
[named-entity-recognition-spacy.md](named-entity-recognition-spacy.md)
{% endcontent-ref %}
