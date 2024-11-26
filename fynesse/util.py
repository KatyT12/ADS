def flatten_tags(tags):
  tags_list = []
  for t in tags:
    if isinstance(tags[t], list):
      tags_list += tags[t]
    else:
      tags_list.append(t)
  return tags_list