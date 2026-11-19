"use client";

type VideoFrameProps = {
  videoId: string;
  title?: string;
};

export function VideoFrame({ videoId, title }: VideoFrameProps) {
  return (
    <div className="video-frame">
      <div className="video-frame__inner">
        <iframe
          title={title ?? "YouTube video"}
          src={`https://www.youtube-nocookie.com/embed/${videoId}?rel=0&modestbranding=1&autoplay=0`}
          className="aspect-video w-full rounded-xl bg-black"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
    </div>
  );
}