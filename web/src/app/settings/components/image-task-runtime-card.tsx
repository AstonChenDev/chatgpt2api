"use client";

import { Clock3, Gauge, LoaderCircle, Save, ShieldCheck } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

import { useSettingsStore } from "../store";

const fields = [
  {
    key: "total_timeout_secs" as const,
    label: "图片及可编辑文件任务总时限",
    defaultValue: 300,
    min: 30,
    max: 1800,
    description:
      "单位秒。图片生成与 PPT/PSD 可编辑文件任务从进入队列开始计时，包含排队、上游处理、下载和保存；默认 300 秒（5 分钟）。",
  },
  {
    key: "max_concurrency" as const,
    label: "全局最大并发",
    defaultValue: 8,
    min: 1,
    max: 32,
    description: "仅控制图片任务。同时执行的图片任务达到上限后进入有界队列，避免耗尽服务线程。",
  },
  {
    key: "max_queue_size" as const,
    label: "最大排队数量",
    defaultValue: 16,
    min: 0,
    max: 256,
    description: "仅控制图片任务。并发已满时最多等待的任务数；设为 0 表示不排队，超出的请求会立即返回繁忙提示。",
  },
  {
    key: "queue_timeout_secs" as const,
    label: "最长排队时间",
    defaultValue: 5,
    min: 0,
    max: 60,
    description: "仅控制图片任务，单位秒。超过该时间仍未开始会快速失败，避免客户端长时间无响应。",
  },
];

export function ImageTaskRuntimeCard() {
  const config = useSettingsStore((state) => state.config);
  const isLoadingConfig = useSettingsStore((state) => state.isLoadingConfig);
  const isSavingConfig = useSettingsStore((state) => state.isSavingConfig);
  const saveConfig = useSettingsStore((state) => state.saveConfig);
  const setImageTaskRuntimeField = useSettingsStore((state) => state.setImageTaskRuntimeField);

  if (isLoadingConfig || !config?.image_task_runtime) {
    return (
      <Card className="rounded-2xl border-white/80 bg-white/90 shadow-sm">
        <CardContent className="flex items-center justify-center p-10">
          <LoaderCircle className="size-5 animate-spin text-stone-400" />
        </CardContent>
      </Card>
    );
  }

  const runtime = config.image_task_runtime;

  return (
    <Card className="rounded-2xl border-white/80 bg-white/90 shadow-sm">
      <CardContent className="space-y-5 p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="flex items-center gap-2 text-base font-semibold text-stone-900">
              <Gauge className="size-5 text-stone-500" />
              图片及可编辑文件任务运行保护
            </div>
            <p className="mt-1 text-xs leading-6 text-stone-500">
              图片生成和 PPT/PSD 共用总时限；图片任务另有可配置的并发与队列保护。
            </p>
          </div>
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-3 py-1 text-xs text-emerald-700">
            <ShieldCheck className="size-3.5" />
            运行保护已启用
          </span>
        </div>

        <div className="flex items-start gap-2 rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-xs leading-6 text-sky-900">
          <Clock3 className="mt-1 size-4 shrink-0" />
          <span>
            推荐保持默认总时限 300 秒。PPT/PSD 使用独立的固定有界运行池（1 个执行、2 个排队）；下方并发与队列设置仅作用于图片任务。
          </span>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          {fields.map((field) => (
            <div key={field.key} className="space-y-2">
              <label className="text-sm text-stone-700" htmlFor={`image-runtime-${field.key}`}>
                {field.label}
              </label>
              <Input
                id={`image-runtime-${field.key}`}
                type="number"
                inputMode="numeric"
                min={field.min}
                max={field.max}
                step={1}
                value={String(runtime[field.key] ?? field.defaultValue)}
                onChange={(event) => setImageTaskRuntimeField(field.key, event.target.value)}
                placeholder={String(field.defaultValue)}
                className="h-10 rounded-xl border-stone-200 bg-white"
              />
              <p className="text-xs leading-5 text-stone-500">{field.description}</p>
            </div>
          ))}
        </div>

        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-6 text-amber-900">
          <p className="font-medium">部署环境变量优先于后台设置</p>
          <p className="mt-1 break-words text-amber-800">
            CHATGPT2API_IMAGE_TASK_TOTAL_TIMEOUT_SECS、CHATGPT2API_IMAGE_TASK_MAX_CONCURRENCY、
            CHATGPT2API_IMAGE_TASK_MAX_QUEUE_SIZE、CHATGPT2API_IMAGE_TASK_QUEUE_TIMEOUT_SECS。
            如已在 Docker 的 .env 中配置，对应输入框保存后不会改变实际生效值。
          </p>
        </div>

        <div className="flex justify-end">
          <Button
            type="button"
            className="h-10 rounded-xl px-5"
            onClick={() => void saveConfig()}
            disabled={isSavingConfig}
          >
            {isSavingConfig ? <LoaderCircle className="size-4 animate-spin" /> : <Save className="size-4" />}
            保存运行保护配置
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
