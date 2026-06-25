import sys
from pathlib import Path

# Add project root using absolute path
sys.path.insert(0, "/Users/astonchen/Documents/chatgpt2api")

from services.config import config
from services.image_storage_service import ImageStorageService

def test_cos_fallback():
    print("=== 开始测试 COS 上传失败降级逻辑 ===")
    
    # Mock ConfigStore.get_image_storage_settings
    mock_settings = {
        "enabled": True,
        "mode": "cos",
        "webdav_url": "",
        "webdav_username": "",
        "webdav_password": "",
        "webdav_root_path": "chatgpt2api/images",
        "cos_secret_id": "invalid_id_for_testing",
        "cos_secret_key": "invalid_key_for_testing",
        "cos_region": "ap-guangzhou",
        "cos_bucket": "invalid-bucket-1250000000",
        "cos_path_prefix": "generated/",
        "public_base_url": ""
    }
    
    # 临时覆盖配置
    original_get_settings = config.get_image_storage_settings
    config.get_image_storage_settings = lambda: mock_settings
    
    try:
        service = ImageStorageService()
        dummy_data = b"PNG dummy image data for testing"
        
        # 调用 save 方法，预期会因为无效 credentials 导致 COS 报错，但能成功降级并返回本地 URL
        stored_image = service.save(dummy_data, base_url="http://localhost:3000")
        
        print("\n测试结果:")
        print(f"返回图片 URL: {stored_image.url}")
        print(f"返回图片 Rel path: {stored_image.rel}")
        print(f"实际存储介质 (storage): {stored_image.storage}")
        
        # 验证返回的是否为本地 URL
        assert "images/" in stored_image.url, f"Expected local image URL, got {stored_image.url}"
        assert stored_image.storage == "local", f"Expected storage to be 'local', got {stored_image.storage}"
        
        # 验证本地文件确实存在
        local_path = Path(service.index_file).parent / "images" / stored_image.rel
        print(f"本地文件路径: {local_path}")
        assert local_path.is_file(), "本地图片文件应该存在"
        
        # 清理测试图片文件
        if local_path.is_file():
            local_path.unlink()
            print("清理测试生成的本地图片成功")
            
        print("🎉 COS 上传失败降级逻辑测试成功！")
        
    finally:
        config.get_image_storage_settings = original_get_settings


def test_list_items_empty_cos_config():
    print("\n=== 开始测试在空 COS 配置下，拉取包含 COS 图片的历史记录不崩溃 ===")
    
    # 模拟空的 COS 配置（比如用户在 local 模式下，没填 COS 字段）
    empty_cos_settings = {
        "enabled": True,
        "mode": "local",
        "cos_secret_id": "",
        "cos_secret_key": "",
        "cos_region": "",  # 为空！
        "cos_bucket": "",
        "cos_path_prefix": "generated/",
        "public_base_url": ""
    }
    
    # 模拟索引中有一条历史 COS 图片记录
    mock_rel = "2026/06/25/123_test.png"
    mock_index = {
        mock_rel: {
            "rel": mock_rel,
            "path": mock_rel,
            "name": "123_test.png",
            "date": "2026-06-25",
            "size": 100,
            "created_at": "2026-06-25 12:00:00",
            "storage": "cos",  # 历史存的是 cos
            "local": False,
            "webdav": False,
            "cos": True,
            "remote_url": "https://test-bucket.cos.ap-guangzhou.myqcloud.com/generated/2026/06/25/123_test.png"
        }
    }
    
    original_get_settings = config.get_image_storage_settings
    config.get_image_storage_settings = lambda: empty_cos_settings
    
    service = ImageStorageService()
    original_load_clean_index = service._load_clean_index
    service._load_clean_index = lambda: mock_index
    
    try:
        # 调用 list_items。如果之前未修复，这里在获取 URL 时初始化 COSClient 会抛出 CosClientError 崩溃。
        items = service.list_items(base_url="http://localhost:3000")
        print("拉取得到的历史记录总数:", len(items))
        
        # 寻找我们模拟的历史 COS 记录
        mock_item_processed = None
        for item in items:
            if item.get("rel") == mock_rel:
                mock_item_processed = item
                break
                
        assert mock_item_processed is not None, "应该在返回的列表中找到模拟的那条 COS 历史图片记录"
        print(f"模拟历史记录的 URL: {mock_item_processed['url']}")
        assert "myqcloud.com" in mock_item_processed["url"], "应该成功生成带有 cos 的域名，而不崩溃"
        
        print("🎉 空 COS 配置下，拉取历史 COS 记录测试成功，无崩溃！")
        
    finally:
        config.get_image_storage_settings = original_get_settings
        service._load_clean_index = original_load_clean_index

if __name__ == "__main__":
    test_cos_fallback()
    test_list_items_empty_cos_config()
