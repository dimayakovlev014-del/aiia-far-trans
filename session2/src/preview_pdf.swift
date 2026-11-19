// Local PDF raster previews for layout review; no model or session-log manipulation.
import AppKit
import PDFKit

guard CommandLine.arguments.count == 3,
      let document = PDFDocument(url: URL(fileURLWithPath: CommandLine.arguments[1])) else {
    fatalError("Usage: swift preview_pdf.swift INPUT.pdf OUTPUT_DIRECTORY")
}
let folder = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
for index in 0..<document.pageCount {
    guard let page = document.page(at: index) else { fatalError("Missing PDF page") }
    let bounds = page.bounds(for: .mediaBox)
    let scale: CGFloat = 1.4
    let width = Int(bounds.width * scale)
    let height = Int(bounds.height * scale)
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setFillColor(CGColor(gray: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.scaleBy(x: scale, y: scale)
    page.draw(with: .mediaBox, to: context)
    let bitmap = NSBitmapImageRep(cgImage: context.makeImage()!)
    let png = bitmap.representation(using: .png, properties: [:])!
    try png.write(to: folder.appendingPathComponent(String(format: "page_%02d.png", index + 1)))
}
print("Rendered \(document.pageCount) PDF pages for AI layout review")
