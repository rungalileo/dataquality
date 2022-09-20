from datasets import  Dataset,ClassLabel, Features, Value

features = Features({
    'text': Value(dtype='string', id=None),
    'label': ClassLabel(num_classes=2, names=['neg', 'pos'])})


mock_dataset = Dataset.from_dict({'text': ['i didnt feel humiliated',
  'i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake',
  'im grabbing a minute to post i feel greedy wrong',
  'i am ever feeling nostalgic about the fireplace i will know that it is still on the property',
  'i am feeling grouchy',
  'ive been feeling a little burdened lately wasnt sure why that was',
  'ive been taking or milligrams or times recommended amount and ive fallen asleep a lot faster but i also feel like so funny',
  'i feel as confused about life as a teenager or as jaded as a year old man',
  'i have been with petronas for years i feel that petronas has performed well and made a huge profit',
  'i feel romantic too',
  'i feel like i have to make the suffering i m seeing mean something',
  'i do feel that running is a divine experience and that i can expect to have some type of spiritual encounter',
  'i think it s the easiest time of year to feel dissatisfied',
  'i feel low energy i m just thirsty',
  'i have immense sympathy with the general point but as a possible proto writer trying to find time to write in the corners of life and with no sign of an agent let alone a publishing contract this feels a little precious',
  'i do not feel reassured anxiety is on each side',
  'i didnt really feel that embarrassed',
  'i feel pretty pathetic most of the time',
  'i started feeling sentimental about dolls i had as a child and so began a collection of vintage barbie dolls from the sixties',
  'i now feel compromised and skeptical of the value of every unit of work i put in'],
 'label': [0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1]},features)