for (new_x, new_y) in stream:
    old_x, old_y = buffer.sample()

    x = concat((new_x, old_x))
    y = concat((new_y, old_y))

    loss = loss_fn(model(x), y)

    optimizer(model, loss)

    buffer.add(new_x, new_y)

for (new_x, new_y) in stream:
    old_x, old_y = buffer.sample()

    new_loss = new_loss_fn(model(new_x), new_y)
    old_loss = old_loss_fn(model(old_x), old_y)

    loss = (new_loss + old_loss) / 2

    optimizer(model, loss)

    buffer.add(new_x, new_y)
    
(Pdb) inc_data['y'].reshape(-1, 1)
tensor([[2],
        [2],
        [3],
        [2],
        [3],
        [3],
        [3],
        [2],
        [2],
        [3]], device='cuda:0')

(Pdb) logits[:, :4]
tensor([[ 4.5505,  9.1793, -1.9217, -2.8466],
        [ 9.1452,  4.9925, -2.5497, -3.3234],
        [ 7.0790,  8.0090, -2.7048, -2.8275],
        [ 9.5243,  4.3168, -2.4265, -3.2403],
        [ 6.1687,  8.9765, -2.4209, -3.2716],
        [ 2.4848,  9.6160, -1.6881, -2.3383],
        [ 8.2123,  7.4308, -2.5390, -3.4883],
        [ 9.2882,  4.0804, -2.4172, -3.0743],
        [ 7.7395,  7.2126, -2.8426, -3.4101],
        [ 8.1175,  6.6969, -2.7293, -3.0066]], device='cuda:0',
       grad_fn=<SliceBackward>)

(Pdb) (mask.int())[:, :4]
tensor([[0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1]], device='cuda:0', dtype=torch.int32)
