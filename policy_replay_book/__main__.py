import argparse
import sys
from .io import read_json, write_json
from .decisions import inspect


def main():
    parser = argparse.ArgumentParser(description='保存智能体在请求确认、直接办理和转人工之间的决策记录，核对曝光概率并汇总实际收益。')
    parser.add_argument('command', choices=['inspect'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()
    try:
        document = read_json(args.input)
        result = inspect(document)
        write_json(args.output, result)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print('处理失败：' + str(error), file=sys.stderr)
        return 2
    print('结果已保存：' + args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
