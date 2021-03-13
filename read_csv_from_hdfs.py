def get_file_list(path_pattern=[], root_path):
    """
    生成hdfs file list
    :param path_pattern:
    :param root_path
    :return:
    """
    cmd = """
        hadoop fs -ls -R {0}
    """.format(root_path)
    if len(path_pattern) == 0:
        pattern = "|".join(["(" + str(p.replace('/', '\/')) + ")" for p in path_pattern])
    else:
        pattern = ""
	
	# 筛选文件
    def validate_path_pattern(path):
        if pattern != "" and re.search(pattern, path) and '_SUCCESS' not in path:
            return True
        elif pattern == "" and '_SUCCESS' not in path:
            return True
        else:
            return False

    status, output = subprocess.getstatusoutput(cmd)
    output = output.split('\n')
    output = list(filter(validate_path_pattern, output))
    file_list = list()
    polluted = any(len(info.split()) != 8 for info in output)
    if status == 0 and len(output) > 0 and not polluted:
        file_list = [info.split()[-1] for info in output if info[0] == '-']
    return file_list

def input_fn(files, batch_size=32, perform_shuffle=False, separator='\t', has_header=False):
    """
    input_fn 用于tf.estimators
    :param files:
    :param batch_size:
    :param perform_shuffle:
    :param separator:
    :param has_header: csv文件是否包含列名
    :return:
    """
    def get_columns(file):
        cmd = """hadoop fs -cat {0} | head -1""".format(file)
        status, output = subprocess.getstatusoutput(cmd)
        return output.split("\n")[0].split(separator)

    def map_fn(line):
        defaults = []
        for col in all_columns:
            if col in CONTINUOUS_COLUMNS + ['label']:
                defaults.append([0.0])
            else:
                defaults.append(['0'])
        columns = tf.compat.v2.io.decode_csv(line, defaults, separator, use_quote_delim=False)

        feature_map = dict()

        for fea, col in zip(all_columns, columns):
            if fea not in USE_COLUMNS:
                continue
            feature_map[fea] = col
        labels = feature_map['label']

        return feature_map, labels

    if has_header:
        all_columns = get_columns(files[0])
        # 使用.skip() 跳过csv的第一行
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.flat_map(lambda filename: (
            tf.data.TextLineDataset(filename).skip(1).map(map_fn)))
    else:
        all_columns = COLUMNS
        dataset = tf.data.TextLineDataset(files).map(map_fn())

    if perform_shuffle:
        dataset = dataset.shuffle(512)
    dataset = dataset.batch(batch_size)
    return dataset

