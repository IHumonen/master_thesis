def df_preparation(df, task='wiki-wiki'):

    df = df.iloc[df['positions'].dropna().index]
    df['gold_sense_id'] = df['gold_sense_id'].apply(str)
    df['positions'] = df['positions'].apply(lambda x: x.split(','))
    df['positions'] = df['positions'].apply(lambda x: x[0].split('-'))
    df[df['context'].apply(lambda x: len(x.split('.'))) != 1]
    df['word_form'] = df.apply(lambda x: get_word_form(x['context'], x['positions'], task), axis=1)

    return df

def get_word_form(context, position, task):
    if task == 'bts-rnc':
        raw = context[int(position[0]): int(position[1])+1]
    else:
        raw = context[int(position[0]): int(position[1])]
    fixed = ''
    for letter in raw:#.lower():
        if letter.isalpha():
            fixed += letter
#             if letter != 'й':
#                 fixed += letter
#             else:
#                 fixed += 'и'

    return fixed

def masking(context, positions, mask_string='<mask>'):
    
    with_mask = ''
    
    for i, symbol in enumerate(context):
        if i == int(positions[0]):
            with_mask += mask_string
        elif int(positions[0]) < i < int(positions[1]):
            pass
        else:
            with_mask += symbol

#only for models woith tokenizers
def get_word_location(target, tokens):
    current = ''
    current_indices = []
    for i, token in enumerate(tokens):
        if token[:2] == '##':
            current += token[2:]
            current_indices.append(i)
        else:
            current = token
            current_indices = [i]
        if current == target:
            return current_indices
    print(target, tokens)
    return 'not found'
            
    return with_mask