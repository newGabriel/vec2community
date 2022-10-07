def read_input_list(file):
    """

    :param file: file name
    :return: list with a listed archives in file
    """
    archives = list()

    try:
        files = open(file, 'r')
        for i in files.readlines():
            archives.append(i[:-1])
    finally:
        pass

    return archives
