def get_phn_seq(phn_boundary, length, reduce_num=2):
    interval = 2**reduce_num
    transform_length = (length+interval-1)//interval

    start = 0
    phn_idx = 0

    all_phn = []
    for i in range(transform_length):
        end = start + interval
        while True:
            if  end >= phn_boundary[phn_idx][1]:
                break
            else:
                phn_idx += 1


