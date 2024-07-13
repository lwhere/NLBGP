import asyncio
import json
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from tqdm.asyncio import tqdm  # 使用tqdm的asyncio支持版本

# 定义GraphQL查询模板
query_template = """
{
asn(asn:"{as_n}") {
    date
    cliqueMember
    asnDegree {
        transit
    }
   }
}
"""

# 创建一个GraphQL客户端
transport = AIOHTTPTransport(url="https://api.asrank.caida.org/v2/graphql")
client = Client(transport=transport, fetch_schema_from_transport=True)

# 获取AS的transit degree


async def get_as_information(asn):
    query = gql(query_template.replace("{as_n}", asn))
    try:
        response = await client.execute_async(query)
        return {"asn": asn, "as_information":response["asn"]}
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"Failed to get as information for ASN {asn}: {e}")
        return None

# 读取JSON文件并提取AS列表


def read_as_numbers(input_path):
    with open(input_path, "r") as file:
        data = json.load(file)
    as_numbers = set()
    for entry in data:
        as_path = entry.get("as_path", "")
        as_numbers.update(as_path.split("-"))
    return as_numbers

# 处理主函数
async def main(input_path):
    as_numbers = read_as_numbers(input_path)
    as_information_list = []

    for asn in tqdm(as_numbers):
        as_information = await get_as_information(asn)
        if as_information is not None:
            as_information_list.append(as_information)

    return as_information_list

# 运行主函数
if __name__ == "__main__":
    input_path = "/Users/hugo/Projects/NLBGP/BGPAgent/filtered_data/bgpleak_different_length/as_path_length_18.json"
    as_information_list = asyncio.run(main(input_path))
    output_path = "/Users/hugo/Projects/NLBGP/BGPAgent/filtered_data/as_information_latest_result.json"
    # 将数据写入到JSON文件中
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(as_information_list, json_file, ensure_ascii=False, indent=4)