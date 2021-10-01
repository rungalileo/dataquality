
import apache_beam as beam
with beam.Pipeline() as p:
    (
        p | beam.Create([1,2,3,4,5])
        | beam.Map(print)
    )
