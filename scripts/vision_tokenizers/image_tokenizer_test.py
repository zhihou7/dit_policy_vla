import torch
from tokenizers_rt1.image_tokenizer import RT1ImageTokenizer

if __name__ == '__main__':
    tokenizer = RT1ImageTokenizer(
        embedding_output_dim=512,
        use_token_learner=True,
        num_tokens=8
    )

    image = torch.rand(2,6,3,300,300)
    context = torch.rand(2,6,512)
    image_tokens = tokenizer(image, context)
    print(image_tokens.shape)