"""Private, single-process 119 demonstration server."""
import base64
import binascii
from functools import lru_cache
import hmac
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import time
import zlib
from urllib.parse import urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError
from store import DEFAULT_WORKERS, Inbox, LEVELS, StoreConflict

ROOT = Path(__file__).resolve().parent
MAX_BODY = 6 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 16_000_000


def thumbnail(data, edge, *, legacy_orientation=False):
    with Image.open(data) as photo:
        if photo.format not in ("JPEG","PNG","WEBP") or photo.width * photo.height > Image.MAX_IMAGE_PIXELS:
            raise ValueError("JPEG·PNG·WebP 사진을 1,600만 화소 이하로 첨부하세요.")
        photo.thumbnail((edge,edge))
        target = (ImageOps.exif_transpose(photo) if legacy_orientation else photo).convert("RGB")
        out = BytesIO()
        target.save(out,"JPEG",quality=85)
        return out.getvalue()


def upload_extension(raw):
    with Image.open(BytesIO(raw)) as photo:
        extensions = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
        if photo.format not in extensions or photo.width * photo.height > Image.MAX_IMAGE_PIXELS:
            raise ValueError("지원하지 않는 사진입니다.")
        photo.load()
        return extensions[photo.format]


@lru_cache(maxsize=192)
def preview(path):
    with open(path,"rb") as fp:
        return thumbnail(fp,400)


def tokens(home):
    path = home/"access.json"
    if not path.exists():
        with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"w") as fp:
            json.dump({role:secrets.token_urlsafe(24) for role in ("report","admin")},fp)
    return json.loads(path.read_text())


def string(data,key,limit):
    val = data.get(key)
    if not isinstance(val,str) or not 0 < len(val.strip()) <= limit:
        raise ValueError("입력 항목과 길이를 확인하세요.")
    return val.strip()


class Error(Exception):
    def __init__(self,status,message):
        self.status = status
        super().__init__(message)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def send(self,code,data=b"",kind="application/json; charset=utf-8",cookie=None):
        if not isinstance(data,bytes):
            data = json.dumps(data,ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type",kind)
        self.send_header("Content-Length",str(len(data)))
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.send_header("Referrer-Policy","no-referrer")
        if cookie:
            self.send_header("Set-Cookie",cookie)
        self.end_headers()
        self.wfile.write(data)

    def role(self, token):
        if isinstance(token,str) and token.isascii():
            for role, secret in self.server.tokens.items():
                if hmac.compare_digest(token,secret):
                    return role
        return None

    def authorize(self, required):
        bearer = self.headers.get("Authorization","")
        if bearer:
            roles = {self.role(bearer.removeprefix("Bearer "))}
        else:
            jar = SimpleCookie()
            jar.load(self.headers.get("Cookie",""))
            roles = {self.role(v.value) for k,v in jar.items() if k in ("demo119_admin","demo119_report")}
        if "admin" not in roles and required not in roles:
            raise Error(403,"이 화면에 맞는 접속 토큰이 필요합니다.")

    def events(self):
        inbox = self.server.inbox
        self.send_response(200)
        self.send_header("Content-Type","text/event-stream")
        self.send_header("Cache-Control","no-cache")
        self.send_header("X-Accel-Buffering","no")
        # Flush each event independently; subsequent snapshots compress against prior cards.
        compressed = "gzip" in self.headers.get("Accept-Encoding", "")
        if compressed:
            encoder = zlib.compressobj(2, zlib.DEFLATED, 31)
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        version = -1
        while True:
            with inbox.changed:
                updated = inbox.changed.wait_for(lambda: inbox.version != version, timeout=12)
                version = inbox.version
                payload = b"event: state\ndata: "+inbox.state()+b"\n\n" if updated else b": ping\n\n"
            if compressed:
                payload = encoder.compress(payload) + encoder.flush(zlib.Z_SYNC_FLUSH)
            self.wfile.write(payload)
            self.wfile.flush()

    def read_json(self):
        if self.headers.get_content_type() != "application/json":
            raise Error(415,"JSON 요청이 필요합니다.")
        n = int(self.headers.get("Content-Length","0"))
        if not 0 < n <= MAX_BODY:
            self.close_connection = True
            raise Error(413,"사진 파일 크기를 줄여 주세요.")
        data = json.loads(self.rfile.read(n))
        if not isinstance(data,dict):
            raise ValueError("객체 형태의 요청이 필요합니다.")
        return data

    def route(self):
        path = urlsplit(self.path).path
        inbox = self.server.inbox
        if self.command == "GET":
            if path == "/health":
                return self.send(200,{"ok":True,"server_ts":time.time()})
            if path.startswith("/api/"):
                self.authorize("admin")
                if path == "/api/state":
                    return self.send(200,inbox.state())
                if path == "/api/events":
                    return self.events()
                match = re.fullmatch(r"/api/reports/([a-f0-9]{32})/image",path)
                if match:
                    with inbox.changed:
                        image = inbox.cards[match[1]]["path"]
                    return self.send(200,preview(image),"image/jpeg")
            else:
                name = {"/":"report.html","/board":"board.html","/board/":"board.html"}.get(path,path.removeprefix("/static/").lstrip("/"))
                if re.fullmatch(r"[a-zA-Z0-9_.-]+",name) and (ROOT/"static"/name).is_file():
                    return self.send(200,(ROOT/"static"/name).read_bytes(),mimetypes.guess_type(name)[0] or "application/octet-stream")
        elif self.command == "POST":
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                raise Error(403,"요청 출처를 확인하세요.")
            if path != "/api/session":
                self.authorize("report" if path == "/api/reports" else "admin")
            data = self.read_json()
            if path == "/api/session":
                token = data.get("token")
                role = self.role(token)
                if not role:
                    raise Error(403,"접속 토큰을 확인하세요.")
                return self.send(200,{"ok":True,"role":role},cookie=f"demo119_{role}={token}; HttpOnly; SameSite=Strict; Path=/")
            request_id = data.get("request_id")
            if request_id is not None and (not isinstance(request_id,str) or not re.fullmatch("[a-zA-Z0-9_-]{8,80}",request_id)):
                raise ValueError("잘못된 요청 ID입니다.")
            if path == "/api/reports":
                text = string(data,"text",1500)
                image = base64.b64decode(string(data,"image",MAX_BODY),validate=True)
                extension = upload_extension(image)
                try:
                    key = inbox.upload(text, image, request_id=request_id, extension=extension)
                except StoreConflict:
                    # Older receipts fingerprinted the server-normalized upload.
                    legacy = thumbnail(BytesIO(image), 1280, legacy_orientation=True)
                    key = inbox.upload(text, legacy, request_id=request_id)
                return self.send(201, {"id": key})
            if path == "/api/example":
                return self.send(201,{"id":inbox.add_examples(string(data,"kind",15),request_id=request_id)[0]})
            if path == "/api/outage-recovery":
                enabled = data.get("enabled")
                if type(enabled) is not bool:
                    raise ValueError("복구 여부는 불리언이어야 합니다.")
                inbox.set_outage_recovery(enabled)
                return self.send(200, {"ok": True, "enabled": enabled})
            if path == "/api/burst":
                return self.send(202,{"ids":inbox.add_examples(request_id=request_id)})
            if path == "/api/rules":
                inbox.rule(string(data,"text",160))
                return self.send(200,{"ok":True})
            match = re.fullmatch(r"/api/reports/([a-f0-9]{32})/label",path)
            if match and data.get("urgency") in LEVELS:
                inbox.label(match[1],data["urgency"],expected_revision=data.get("revision"))
                return self.send(200,{"ok":True})
        raise Error(404,"요청한 항목이 없습니다.")

    def dispatch(self):
        try:
            self.route()
        except (BrokenPipeError,ConnectionResetError):
            return
        except Error as e:
            self.close_connection = True
            self.send(e.status,{"error":str(e)})
        except StoreConflict as e:
            self.send(409,{"error":str(e)})
        except KeyError:
            self.send(404,{"error":"대상을 찾을 수 없습니다."})
        except (ValueError,binascii.Error,UnidentifiedImageError,Image.DecompressionBombError,OSError):
            self.send(400,{"error":"입력과 사진 형식을 확인하세요."})

    do_GET = dispatch
    do_POST = dispatch


def main():
    os.umask(0o077)
    home = Path(os.environ.get("DEMO_HOME",ROOT/"runtime")).resolve()
    home.mkdir(parents=True,exist_ok=True)
    host = "127.0.0.1"
    port = int(os.environ.get("PORT","8119"))
    server = ThreadingHTTPServer((host,port),Handler)
    server.tokens = tokens(home)
    workers = int(os.environ.get("DEMO_WORKERS",DEFAULT_WORKERS))
    server.inbox = Inbox(home,os.environ.get("CRISIS_ROOT","/ephemeral/crisis_bench/crisismmd"),
                         ROOT/"fixtures/reports.json",workers)
    print(f"119 demo ready http://{host}:{port} | runtime {home} | workers {workers}",flush=True)
    server.serve_forever()


if __name__=="__main__":
    main()
